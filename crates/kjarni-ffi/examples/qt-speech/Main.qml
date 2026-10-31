import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import KjarniSpeech

ApplicationWindow {
    width: 640
    height: 420
    visible: true
    title: "Local speech to text"

    SpeechToText { id: speech }

    FileDialog {
        id: fileDialog
        nameFilters: ["Audio (*.wav *.mp3 *.flac *.ogg)"]
        onAccepted: speech.transcribeFile(selectedFile)
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            ComboBox {
                Layout.fillWidth: true
                model: speech.microphones
                currentIndex: speech.microphone
                enabled: !speech.recording
                onActivated: (index) => speech.microphone = index
            }
            Button {
                text: speech.recording ? "Stop" : "Record"
                enabled: speech.ready && !speech.busy
                onClicked: speech.recording ? speech.stopRecording() : speech.startRecording()
            }
            Button {
                text: "Open file..."
                enabled: speech.ready && !speech.busy && !speech.recording
                onClicked: fileDialog.open()
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            TextArea {
                text: speech.transcript
                readOnly: true
                wrapMode: TextEdit.Wrap
                placeholderText: "The transcript appears here."
            }
        }

        RowLayout {
            BusyIndicator {
                running: speech.busy || !speech.ready
                visible: running
                implicitWidth: 24
                implicitHeight: 24
            }
            Label {
                Layout.fillWidth: true
                text: speech.status
                elide: Text.ElideRight
            }
        }
    }
}
