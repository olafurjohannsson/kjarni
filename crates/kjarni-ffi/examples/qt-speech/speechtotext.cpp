#include "speechtotext.h"

#include <QAudioDevice>
#include <QElapsedTimer>
#include <QFuture>
#include <QMediaDevices>
#include <QtConcurrent>

#include <span>

namespace {

struct Loaded {
    std::shared_ptr<kjarni::Transcriber> transcriber;
    QString error;
};

Loaded load() {
    auto transcriber = kjarni::Transcriber::create({.model = "whisper-small"});
    if (!transcriber) {
        return {nullptr, QString::fromStdString(transcriber.error().message())};
    }
    return {std::make_shared<kjarni::Transcriber>(std::move(*transcriber)), {}};
}

QString timestamp(float seconds) {
    const int s = static_cast<int>(seconds);
    return QString("%1:%2").arg(s / 60, 2, 10, QChar('0')).arg(s % 60, 2, 10, QChar('0'));
}

QString formatted(const kjarni::Transcript& t) {
    QString out;
    for (const kjarni::Segment& s : t.segments) {
        out += QString("[%1] %2\n").arg(timestamp(s.start), QString::fromStdString(s.text).trimmed());
    }
    return out;
}

// Asks for floats at the device's own rate and channel count. Whisper's 16 kHz is left
// to Kjarni, which resamples, because many devices refuse to open at that rate.
QAudioFormat captureFormat(const QAudioDevice& device) {
    QAudioFormat format = device.preferredFormat();
    format.setSampleFormat(QAudioFormat::Float);
    return device.isFormatSupported(format) ? format : device.preferredFormat();
}

}  // namespace

SpeechToText::SpeechToText(QObject* parent) : QObject(parent) {
    QtConcurrent::run(load).then(this, [this](Loaded loaded) {
        if (!loaded.transcriber) {
            setStatus("Could not load the model: " + loaded.error);
            return;
        }
        m_transcriber = std::move(loaded.transcriber);
        setStatus("Ready. Nothing leaves this machine.");
    });
}

QStringList SpeechToText::microphones() const {
    QStringList names;
    for (const QAudioDevice& device : QMediaDevices::audioInputs()) {
        names << device.description();
    }
    return names;
}

void SpeechToText::setMicrophone(int index) {
    if (index != m_microphone) {
        m_microphone = index;
        emit microphoneChanged();
    }
}

void SpeechToText::startRecording() {
    const auto devices = QMediaDevices::audioInputs();
    if (m_source || m_busy || !ready() || m_microphone >= devices.size()) {
        return;
    }
    const QAudioDevice& device = devices[m_microphone];
    m_format = captureFormat(device);
    m_samples.clear();
    m_source = std::make_unique<QAudioSource>(device, m_format);
    m_stream = m_source->start();
    connect(m_stream, &QIODevice::readyRead, this, &SpeechToText::capture);
    setStatus("Recording...");
}

// Mixes whatever the device delivers down to mono floats, which is what Whisper takes.
void SpeechToText::capture() {
    const QByteArray bytes = m_stream->readAll();
    const int channels = m_format.channelCount();
    const int frameSize = m_format.bytesPerFrame();
    for (qsizetype frame = 0; frame + frameSize <= bytes.size(); frame += frameSize) {
        float mono = 0.0f;
        for (int c = 0; c < channels; ++c) {
            mono += m_format.normalizedSampleValue(bytes.constData() + frame + c * m_format.bytesPerSample());
        }
        m_samples.push_back(mono / channels);
    }
}

void SpeechToText::stopRecording() {
    if (!m_source) {
        return;
    }
    capture();
    m_source->stop();
    m_source.reset();
    m_stream = nullptr;
    transcribeSamples();
}

void SpeechToText::transcribeSamples() {
    const auto rate = static_cast<std::uint32_t>(m_format.sampleRate());
    run([samples = std::move(m_samples), rate](const kjarni::Transcriber& t) {
        return t.transcribe(std::span<const float>(samples), rate);
    });
    m_samples.clear();
}

void SpeechToText::transcribeFile(const QUrl& file) {
    if (m_source || m_busy || !ready()) {
        return;
    }
    run([path = file.toLocalFile().toStdString()](const kjarni::Transcriber& t) {
        return t.transcribe_file(path);
    });
}

// The transcriber stays alive through the shared_ptr captured by the worker, and `m_busy`
// keeps a second call off the handle, which serves one request at a time.
void SpeechToText::run(std::function<kjarni::Result<kjarni::Transcript>(const kjarni::Transcriber&)> work) {
    m_busy = true;
    setStatus("Transcribing...");

    struct Outcome {
        kjarni::Result<kjarni::Transcript> result;
        qint64 elapsedMs;
    };
    QtConcurrent::run([transcriber = m_transcriber, work = std::move(work)] {
        QElapsedTimer timer;
        timer.start();
        auto result = work(*transcriber);
        return std::make_shared<Outcome>(Outcome{std::move(result), timer.elapsed()});
    }).then(this, [this](std::shared_ptr<Outcome> outcome) {
        m_busy = false;
        if (!outcome->result) {
            setStatus("Transcription failed: " + QString::fromStdString(outcome->result.error().message()));
            return;
        }
        m_transcript = formatted(*outcome->result);
        emit transcriptChanged();
        setStatus(QString("%1 s of audio in %2 s, language: %3")
                      .arg(outcome->result->duration_secs, 0, 'f', 1)
                      .arg(outcome->elapsedMs / 1000.0, 0, 'f', 1)
                      .arg(QString::fromStdString(outcome->result->language)));
    });
}

void SpeechToText::setStatus(const QString& status) {
    m_status = status;
    emit stateChanged();
}
