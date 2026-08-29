// Kjarni exposed to QML.
//
// Inference is compute-bound and takes anywhere from milliseconds to seconds. Doing
// it on Qt's GUI thread freezes the window, so every call here is dispatched to a
// worker via QtConcurrent and reported back through signals. QML sees an ordinary
// asynchronous object: call a method, connect to a signal.
//
//   KjarniEmbedder {
//       id: embedder
//       model: "minilm-l6-v2"
//       onReady: embedder.rank("refund", ["...", "..."])
//       onRanked: (results) => listModel.update(results)
//   }

#pragma once

#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QtConcurrent>

#include <memory>
#include <optional>

#include "kjarni.hpp"

/// Semantic similarity, exposed to QML.
class KjarniEmbedder : public QObject {
    Q_OBJECT
    QML_ELEMENT

    /// Model name from the registry. Setting it reloads, asynchronously.
    Q_PROPERTY(QString model READ model WRITE setModel NOTIFY modelChanged)
    /// True once the model has loaded and calls will succeed.
    Q_PROPERTY(bool ready READ isReady NOTIFY readyChanged)
    /// True while a load or a query is in flight, for binding to a BusyIndicator.
    Q_PROPERTY(bool busy READ isBusy NOTIFY busyChanged)

public:
    explicit KjarniEmbedder(QObject* parent = nullptr) : QObject(parent) {}

    [[nodiscard]] QString model() const { return model_; }
    [[nodiscard]] bool isReady() const { return embedder_.has_value(); }
    [[nodiscard]] bool isBusy() const { return busy_; }

    void setModel(const QString& name) {
        if (name == model_) return;
        model_ = name;
        emit modelChanged();
        load();
    }

    /// Loads the model. Safe to call from QML's Component.onCompleted.
    Q_INVOKABLE void load() {
        if (busy_) return;
        setBusy(true);

        const std::string name = model_.toStdString();
        auto future = QtConcurrent::run([name] {
            return kjarni::Embedder::create({.model = name});
        });

        watch(future, [this](kjarni::Result<kjarni::Embedder>&& result) {
            if (!result) {
                setBusy(false);
                emit failed(QString::fromStdString(result.error().message()));
                return;
            }
            embedder_.emplace(std::move(*result));
            setBusy(false);
            emit readyChanged();
            emit ready();
        });
    }

    /// Ranks `documents` against `query`, best first.
    ///
    /// Emits `ranked` with a list of `{ index, text, score }` objects, which a
    /// ListView can consume directly.
    Q_INVOKABLE void rank(const QString& query, const QStringList& documents) {
        if (!embedder_ || busy_) return;
        setBusy(true);

        std::vector<std::string> texts;
        texts.reserve(documents.size() + 1);
        texts.push_back(query.toStdString());
        for (const auto& d : documents) texts.push_back(d.toStdString());

        auto* embedder = &*embedder_;
        auto future = QtConcurrent::run([embedder, texts = std::move(texts)] {
            return embedder->encode(texts);
        });

        watch(future, [this, documents](kjarni::Result<std::vector<std::vector<float>>>&& r) {
            setBusy(false);
            if (!r) {
                emit failed(QString::fromStdString(r.error().message()));
                return;
            }

            const auto& vectors = *r;
            QVariantList out;
            for (int i = 0; i < documents.size(); ++i) {
                QVariantMap row;
                row["index"] = i;
                row["text"] = documents[i];
                row["score"] = kjarni::cosine(vectors[0], vectors[i + 1]);
                out.append(row);
            }

            std::sort(out.begin(), out.end(), [](const QVariant& a, const QVariant& b) {
                return a.toMap()["score"].toFloat() > b.toMap()["score"].toFloat();
            });

            emit ranked(out);
        });
    }

signals:
    void modelChanged();
    void readyChanged();
    void busyChanged();
    /// The model finished loading.
    void ready();
    /// Results, sorted by score descending.
    void ranked(const QVariantList& results);
    /// Anything went wrong, with the library's own message.
    void failed(const QString& message);

private:
    void setBusy(bool value) {
        if (busy_ == value) return;
        busy_ = value;
        emit busyChanged();
    }

    /// Delivers a QFuture's result on the object's own thread.
    ///
    /// QtConcurrent runs the work on a pool thread; touching QObject state from
    /// there would be a data race. The watcher marshals it back.
    template <typename T, typename F>
    void watch(QFuture<T> future, F&& handler) {
        auto* watcher = new QFutureWatcher<T>(this);
        connect(watcher, &QFutureWatcherBase::finished, this,
                [watcher, handler = std::forward<F>(handler)]() mutable {
                    handler(std::move(watcher->result()));
                    watcher->deleteLater();
                });
        watcher->setFuture(future);
    }

    QString model_ = QStringLiteral("minilm-l6-v2");
    std::optional<kjarni::Embedder> embedder_;
    bool busy_ = false;
};

/// A local language model, exposed to QML with token streaming.
class KjarniChat : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(QString model READ model WRITE setModel NOTIFY modelChanged)
    Q_PROPERTY(bool ready READ isReady NOTIFY readyChanged)
    Q_PROPERTY(bool generating READ isGenerating NOTIFY generatingChanged)

public:
    explicit KjarniChat(QObject* parent = nullptr) : QObject(parent) {}

    [[nodiscard]] QString model() const { return model_; }
    [[nodiscard]] bool isReady() const { return chat_.has_value(); }
    [[nodiscard]] bool isGenerating() const { return generating_; }

    void setModel(const QString& name) {
        if (name == model_) return;
        model_ = name;
        emit modelChanged();
        load();
    }

    Q_INVOKABLE void load() {
        if (generating_) return;

        const std::string name = model_.toStdString();
        auto future = QtConcurrent::run([name] {
            return kjarni::Chat::create({.model = name});
        });

        auto* watcher = new QFutureWatcher<kjarni::Result<kjarni::Chat>>(this);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            auto result = std::move(watcher->result());
            watcher->deleteLater();
            if (!result) {
                emit failed(QString::fromStdString(result.error().message()));
                return;
            }
            chat_.emplace(std::move(*result));
            emit readyChanged();
            emit ready();
        });
        watcher->setFuture(future);
    }

    /// Generates a reply, emitting `token` for each fragment as it arrives.
    ///
    /// The callback fires on a worker thread, so it queues each token onto this
    /// object's thread before emitting; QML must never be touched from the pool.
    Q_INVOKABLE void send(const QString& message, int maxTokens = 256) {
        if (!chat_ || generating_) return;
        setGenerating(true);

        auto* chat = &*chat_;
        const std::string prompt = message.toStdString();

        auto future = QtConcurrent::run([this, chat, prompt, maxTokens] {
            auto gen = kjarni::Generation::greedy(maxTokens);
            return chat->stream(prompt, gen, [this](std::string_view piece) {
                const QString text = QString::fromUtf8(piece.data(),
                                                       static_cast<int>(piece.size()));
                QMetaObject::invokeMethod(this, [this, text] { emit token(text); },
                                          Qt::QueuedConnection);
                return true;   // return false here to stop generation early
            });
        });

        auto* watcher = new QFutureWatcher<std::expected<void, kjarni::Error>>(this);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            auto result = std::move(watcher->result());
            watcher->deleteLater();
            setGenerating(false);
            if (!result) {
                emit failed(QString::fromStdString(result.error().message()));
                return;
            }
            emit finished();
        });
        watcher->setFuture(future);
    }

signals:
    void modelChanged();
    void readyChanged();
    void generatingChanged();
    void ready();
    /// One fragment of the response.
    void token(const QString& text);
    /// Generation completed normally.
    void finished();
    void failed(const QString& message);

private:
    void setGenerating(bool value) {
        if (generating_ == value) return;
        generating_ = value;
        emit generatingChanged();
    }

    QString model_ = QStringLiteral("llama3.2-1b-instruct");
    std::optional<kjarni::Chat> chat_;
    bool generating_ = false;
};
