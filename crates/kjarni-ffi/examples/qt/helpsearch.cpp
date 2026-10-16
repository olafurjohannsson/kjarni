#include "helpsearch.h"

#include "articles.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVariantMap>
#include <QtConcurrent/QtConcurrentRun>

#include <string>
#include <vector>

namespace {

constexpr const char* kModel = "minilm-l6-v2";

QString toQString(std::string_view s) {
    return QString::fromUtf8(s.data(), static_cast<qsizetype>(s.size()));
}

QString lastError() {
    return QString::fromUtf8(kjarni_last_error_message());
}

// The text each article is indexed and reranked as.
std::string articleText(const Article& a) {
    return std::string(a.title) + "\n\n" + std::string(a.body);
}

QVariantMap row(std::size_t article, int rank, std::optional<float> score) {
    const Article& a = kArticles[article];
    QVariantMap r;
    r["article"] = static_cast<int>(article);
    r["rank"] = rank;
    r["category"] = toQString(a.category);
    r["title"] = toQString(a.title);
    r["body"] = toQString(a.body);
    r["score"] = score ? QVariant(*score) : QVariant();
    r["previousRank"] = -1;
    return r;
}

}  // namespace

HelpSearch::HelpSearch(QObject* parent) : QObject(parent) {
    showAll();
    setStatus("Indexing articles…");

    // Indexing loads the embedding model (downloading it the first time) and
    // encodes every article: seconds, not milliseconds. Do it on a worker thread
    // and hand the finished searcher back to this object's thread.
    QtConcurrent::run([] { return std::make_shared<Loaded>(load()); })
        .then(this, [this](std::shared_ptr<Loaded> loaded) {
            if (!loaded->searcher) {
                setStatus("Could not build the index: " + loaded->error);
                return;
            }
            m_dir = std::move(loaded->dir);
            m_indexPath = loaded->indexPath;
            m_searcher = std::move(loaded->searcher);
            setStatus(QString("%1 articles indexed").arg(kArticles.size()));
            emit readyChanged();
            run();
        });
}

HelpSearch::Loaded HelpSearch::load() {
    Loaded out;
    out.dir = std::make_unique<QTemporaryDir>();
    if (!out.dir->isValid()) {
        out.error = "no temporary directory";
        return out;
    }

    // The indexer reads files, so each article becomes one. The file name carries
    // the article's position, which is how a search result finds its way back.
    const QString articles = out.dir->filePath("articles");
    QDir().mkpath(articles);
    for (std::size_t i = 0; i < kArticles.size(); ++i) {
        QFile f(QString("%1/%2.md").arg(articles).arg(i, 2, 10, QChar('0')));
        if (!f.open(QIODevice::WriteOnly)) {
            out.error = "could not write " + f.fileName();
            return out;
        }
        f.write(QByteArray::fromStdString(articleText(kArticles[i])));
    }

    KjarniIndexerConfig icfg = kjarni_indexer_config_default();
    icfg.model_name = kModel;
    icfg.quiet = 1;
    KjarniIndexer* indexer = nullptr;
    if (kjarni_indexer_new(&icfg, &indexer) != KJARNI_ERROR_CODE_OK) {
        out.error = lastError();
        return out;
    }

    out.indexPath = out.dir->filePath("help.idx");
    const QByteArray indexPath = out.indexPath.toUtf8();
    const QByteArray input = articles.toUtf8();
    const char* inputs[] = {input.constData()};
    KjarniIndexStats stats{};
    const auto rc = kjarni_indexer_create(indexer, indexPath.constData(), inputs, 1, 1, &stats);
    kjarni_indexer_free(indexer);
    if (rc != KJARNI_ERROR_CODE_OK) {
        out.error = lastError();
        return out;
    }

    KjarniSearcherConfig scfg = kjarni_searcher_config_default();
    scfg.model_name = kModel;
    scfg.quiet = 1;
    KjarniSearcher* searcher = nullptr;
    if (kjarni_searcher_new(&scfg, &searcher) != KJARNI_ERROR_CODE_OK) {
        out.error = lastError();
        return out;
    }
    out.searcher.reset(searcher);
    return out;
}

void HelpSearch::setMode(Mode mode) {
    if (mode == m_mode) return;
    m_mode = mode;
    emit modeChanged();
    run();
}

void HelpSearch::search(const QString& query) {
    m_query = query.trimmed();
    run();
}

void HelpSearch::showAll() {
    m_results.clear();
    for (std::size_t i = 0; i < kArticles.size(); ++i)
        m_results.append(row(i, static_cast<int>(i) + 1, std::nullopt));
    m_reranked = false;
    m_elapsedMs = 0;
    emit resultsChanged();
}

void HelpSearch::run() {
    if (!m_searcher) return;
    if (m_query.isEmpty()) {
        showAll();
        return;
    }

    KjarniSearchOptions opts = kjarni_search_options_default();
    opts.mode = m_mode;
    opts.top_k = kArticles.size();

    // A query embeds one short string and scores 24 articles: a few milliseconds,
    // so it runs on every keystroke on the UI thread.
    QElapsedTimer timer;
    timer.start();
    const QByteArray index = m_indexPath.toUtf8();
    const QByteArray query = m_query.toUtf8();
    KjarniSearchResults found{};
    if (kjarni_searcher_search_with_options(m_searcher.get(), index.constData(),
                                            query.constData(), &opts,
                                            &found) != KJARNI_ERROR_CODE_OK) {
        setStatus("Search failed: " + lastError());
        return;
    }
    m_elapsedMs = timer.nsecsElapsed() / 1e6;

    m_results.clear();
    for (std::size_t i = 0; i < found.len; ++i) {
        const KjarniSearchResult& r = found.results[i];
        const auto meta = QJsonDocument::fromJson(QByteArray(r.metadata_json)).object();
        bool ok = false;
        const auto article = QFileInfo(meta["source"].toString()).baseName().toUInt(&ok);
        if (!ok || article >= kArticles.size()) continue;
        m_results.append(row(article, m_results.size() + 1, r.score));
    }
    kjarni_search_results_free(found);

    m_reranked = false;
    emit resultsChanged();
}

void HelpSearch::rerank() {
    if (m_query.isEmpty() || m_results.isEmpty() || m_reranked || m_rerankerLoading) return;

    std::vector<std::string> docs;
    std::vector<QVariantMap> rows;
    for (const QVariant& v : std::as_const(m_results)) {
        rows.push_back(v.toMap());
        docs.push_back(articleText(kArticles[rows.back()["article"].toUInt()]));
    }

    // The cross-encoder reads the query and each article together, which is much
    // more precise than comparing two vectors and much slower: tens of milliseconds
    // here, and it is a second model to load the first time. So it runs on a worker,
    // over what retrieval already found rather than over everything.
    m_rerankerLoading = true;
    emit rerankerLoadingChanged();

    struct Outcome {
        std::vector<kjarni::Ranked> ranked;
        double ms = 0;
        QString error;
    };
    auto reranker = m_reranker;  // shared, so the worker can create it on first use
    QtConcurrent::run([reranker, query = m_query.toStdString(), docs = std::move(docs)] {
        auto out = std::make_shared<Outcome>();
        if (!*reranker) {
            auto created = kjarni::Reranker::create();
            if (!created) {
                out->error = QString::fromStdString(created.error().message());
                return out;
            }
            reranker->emplace(std::move(*created));
        }
        QElapsedTimer timer;
        timer.start();
        auto ranked = (*reranker)->rerank(query, docs);
        if (!ranked) {
            out->error = QString::fromStdString(ranked.error().message());
            return out;
        }
        out->ms = timer.nsecsElapsed() / 1e6;
        out->ranked = std::move(*ranked);
        return out;
    }).then(this, [this, rows = std::move(rows), query = m_query, mode = m_mode](std::shared_ptr<Outcome> out) {
        m_rerankerLoading = false;
        emit rerankerLoadingChanged();
        if (!out->error.isEmpty()) {
            setStatus("Rerank failed: " + out->error);
            return;
        }
        // The user may have typed or switched mode while this ran.
        if (query != m_query || mode != m_mode) return;

        m_results.clear();
        for (const kjarni::Ranked& r : out->ranked) {
            QVariantMap moved = rows[r.index];
            moved["previousRank"] = moved["rank"];
            moved["rank"] = m_results.size() + 1;
            moved["score"] = r.score;
            m_results.append(moved);
        }
        m_elapsedMs = out->ms;
        m_reranked = true;
        emit resultsChanged();
    });
}

void HelpSearch::setStatus(const QString& status) {
    if (status == m_status) return;
    m_status = status;
    emit statusChanged();
}
