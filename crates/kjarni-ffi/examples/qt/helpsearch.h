#pragma once

#include <QObject>
#include <QString>
#include <QTemporaryDir>
#include <QVariantList>
#include <QtQml/qqmlregistration.h>

#include <memory>
#include <optional>

#include "kjarni.h"
#include "kjarni.hpp"

// Exposes a Kjarni index of the help centre to QML. The articles are indexed once
// at startup, keyword (BM25) and vector together, and then searched in whichever
// mode the user picks. A cross-encoder can rerank the current results on request.
class HelpSearch : public QObject {
    Q_OBJECT
    QML_ELEMENT

public:
    // Same values as KjarniSearchMode, so they pass straight through.
    enum Mode { Keyword = KJARNI_SEARCH_MODE_KEYWORD,
                Semantic = KJARNI_SEARCH_MODE_SEMANTIC,
                Hybrid = KJARNI_SEARCH_MODE_HYBRID };
    Q_ENUM(Mode)

private:
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(Mode mode READ mode WRITE setMode NOTIFY modeChanged)
    Q_PROPERTY(QVariantList results READ results NOTIFY resultsChanged)
    Q_PROPERTY(bool reranked READ reranked NOTIFY resultsChanged)
    Q_PROPERTY(bool rerankerLoading READ rerankerLoading NOTIFY rerankerLoadingChanged)
    Q_PROPERTY(double elapsedMs READ elapsedMs NOTIFY resultsChanged)

public:
    explicit HelpSearch(QObject* parent = nullptr);

    bool ready() const { return m_searcher != nullptr; }
    QString status() const { return m_status; }
    Mode mode() const { return m_mode; }
    void setMode(Mode mode);
    QVariantList results() const { return m_results; }
    bool reranked() const { return m_reranked; }
    bool rerankerLoading() const { return m_rerankerLoading; }
    double elapsedMs() const { return m_elapsedMs; }

    Q_INVOKABLE void search(const QString& query);
    Q_INVOKABLE void rerank();

signals:
    void readyChanged();
    void statusChanged();
    void modeChanged();
    void resultsChanged();
    void rerankerLoadingChanged();

private:
    using SearcherPtr = std::unique_ptr<KjarniSearcher, decltype(&kjarni_searcher_free)>;

    struct Loaded {
        std::unique_ptr<QTemporaryDir> dir;
        QString indexPath;
        SearcherPtr searcher{nullptr, &kjarni_searcher_free};
        QString error;
    };
    static Loaded load();

    void run();
    void showAll();
    void setStatus(const QString& status);

    std::unique_ptr<QTemporaryDir> m_dir;
    QString m_indexPath;
    SearcherPtr m_searcher{nullptr, &kjarni_searcher_free};
    // Created and used on a worker thread, one rerank at a time.
    std::shared_ptr<std::optional<kjarni::Reranker>> m_reranker =
        std::make_shared<std::optional<kjarni::Reranker>>();

    QString m_query;
    Mode m_mode = Hybrid;
    QString m_status;
    QVariantList m_results;
    bool m_reranked = false;
    bool m_rerankerLoading = false;
    double m_elapsedMs = 0;
};
