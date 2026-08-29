// Semantic search and local chat in QML.
//
// Nothing here knows it is talking to a transformer: the model loads in the
// background, results arrive as signals, and `busy` drives the spinner. That is the
// whole point of doing the threading in KjarniModel.h.

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Kjarni

ApplicationWindow {
    id: window
    width: 720
    height: 640
    visible: true
    title: "Kjarni · local inference in Qt"

    KjarniEmbedder {
        id: embedder
        model: "minilm-l6-v2"

        Component.onCompleted: load()
        onReady: status.text = "Model ready. Nothing leaves this machine."
        onFailed: (message) => status.text = "Error: " + message
        onRanked: (results) => {
            hits.clear()
            for (const r of results) hits.append(r)
        }
    }

    KjarniChat {
        id: chat
        model: "llama3.2-1b-instruct"

        onToken: (text) => reply.text += text
        onFailed: (message) => status.text = "Chat error: " + message
    }

    ListModel { id: hits }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        RowLayout {
            Label {
                id: status
                text: "Loading model…"
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            BusyIndicator {
                running: embedder.busy || chat.generating
                implicitWidth: 24
                implicitHeight: 24
            }
        }

        // ── Semantic search ──────────────────────────────────────
        GroupBox {
            title: "Rank by meaning"
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent

                RowLayout {
                    TextField {
                        id: query
                        Layout.fillWidth: true
                        placeholderText: "Query"
                        text: "How do I get my money back?"
                        onAccepted: rankButton.clicked()
                    }
                    Button {
                        id: rankButton
                        text: "Rank"
                        enabled: embedder.ready && !embedder.busy
                        onClicked: embedder.rank(query.text, [
                            "What is your refund policy?",
                            "The delivery arrived three days late.",
                            "Can I return this item for a refund?",
                            "Our office is open until 5pm on weekdays.",
                            "I would like to cancel my subscription.",
                        ])
                    }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 160
                    clip: true
                    model: hits

                    delegate: RowLayout {
                        width: ListView.view.width
                        Label {
                            text: model.text
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Label {
                            text: model.score.toFixed(4)
                            font.family: "monospace"
                            color: index === 0 ? "#3fb950" : "#8b949e"
                        }
                    }
                }
            }
        }

        // ── Chat ─────────────────────────────────────────────────
        GroupBox {
            title: "Chat"
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent

                RowLayout {
                    TextField {
                        id: prompt
                        Layout.fillWidth: true
                        placeholderText: "Ask something"
                        text: "What is the capital of Iceland?"
                        onAccepted: sendButton.clicked()
                    }
                    Button {
                        id: sendButton
                        text: "Send"
                        enabled: chat.ready && !chat.generating
                        onClicked: {
                            reply.text = ""
                            chat.send(prompt.text, 128)
                        }
                    }
                    Button {
                        text: "Load model"
                        visible: !chat.ready
                        onClicked: chat.load()
                    }
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    // Tokens append here as they arrive, so text grows live.
                    TextArea {
                        id: reply
                        readOnly: true
                        wrapMode: TextArea.Wrap
                        placeholderText: "The reply streams in token by token."
                    }
                }
            }
        }
    }
}
