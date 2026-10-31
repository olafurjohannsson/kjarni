#pragma once

#include <QAudioFormat>
#include <QAudioSource>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QUrl>
#include <QtQml/qqmlregistration.h>

#include <functional>
#include <memory>
#include <vector>

#include "kjarni.hpp"

// Exposes Whisper to QML. The model loads once at startup, recording happens on the
// GUI thread through Qt's audio callbacks, and transcription runs on the thread pool.
class SpeechToText : public QObject {
    Q_OBJECT
    QML_ELEMENT
    Q_PROPERTY(bool ready READ ready NOTIFY stateChanged)
    Q_PROPERTY(bool recording READ recording NOTIFY stateChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY stateChanged)
    Q_PROPERTY(QString status READ status NOTIFY stateChanged)
    Q_PROPERTY(QString transcript READ transcript NOTIFY transcriptChanged)
    Q_PROPERTY(QStringList microphones READ microphones CONSTANT)
    Q_PROPERTY(int microphone READ microphone WRITE setMicrophone NOTIFY microphoneChanged)

public:
    explicit SpeechToText(QObject* parent = nullptr);

    bool ready() const { return m_transcriber != nullptr; }
    bool recording() const { return m_source != nullptr; }
    bool busy() const { return m_busy; }
    QString status() const { return m_status; }
    QString transcript() const { return m_transcript; }
    QStringList microphones() const;
    int microphone() const { return m_microphone; }
    void setMicrophone(int index);

    Q_INVOKABLE void startRecording();
    Q_INVOKABLE void stopRecording();
    Q_INVOKABLE void transcribeFile(const QUrl& file);

private:
    void capture();
    void transcribeSamples();
    void run(std::function<kjarni::Result<kjarni::Transcript>(const kjarni::Transcriber&)> work);
    void setStatus(const QString& status);

    std::shared_ptr<kjarni::Transcriber> m_transcriber;
    std::unique_ptr<QAudioSource> m_source;
    QIODevice* m_stream = nullptr;
    QAudioFormat m_format;
    std::vector<float> m_samples;
    int m_microphone = 0;
    bool m_busy = false;
    QString m_status = "Loading the model...";
    QString m_transcript;

signals:
    void stateChanged();
    void transcriptChanged();
    void microphoneChanged();
};
