import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import KjarniSearch

ApplicationWindow {
    id: window
    width: 960
    height: 680
    visible: true
    title: "Help centre search: Kjarni + Qt Quick"

    HelpSearch { id: search }

    // Scores mean different things per mode (BM25 is unbounded, hybrid is a fused
    // rank score, the reranker returns negative logits), so bars are scaled between
    // the worst and best result in the current list.
    readonly property var scoreRange: {
        let lo = Infinity, hi = -Infinity
        for (const r of search.results)
            if (r.score !== undefined) { lo = Math.min(lo, r.score); hi = Math.max(hi, r.score) }
        return { lo: lo, hi: hi }
    }
    function barFraction(score) {
        const span = scoreRange.hi - scoreRange.lo
        return span > 0 ? (score - scoreRange.lo) / span : 1
    }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            spacing: 12

            TextField {
                id: query
                Layout.fillWidth: true
                font.pixelSize: 16
                placeholderText: "Describe the problem in your own words"
                enabled: search.ready
                focus: true
                onTextChanged: search.search(text)
            }

            ButtonGroup { id: modes }
            Repeater {
                model: [
                    { label: "Keyword", mode: HelpSearch.Keyword, tip: "BM25: matches the words you typed" },
                    { label: "Hybrid", mode: HelpSearch.Hybrid, tip: "BM25 and vectors, fused" },
                    { label: "Semantic", mode: HelpSearch.Semantic, tip: "Vectors: matches meaning" },
                ]
                delegate: Button {
                    required property var modelData
                    text: modelData.label
                    checkable: true
                    checked: search.mode === modelData.mode
                    ButtonGroup.group: modes
                    onClicked: search.mode = modelData.mode
                    ToolTip.visible: hovered
                    ToolTip.text: modelData.tip
                }
            }

            Button {
                text: search.rerankerLoading ? "Reranking…" : "Rerank"
                enabled: search.ready && query.text.trim().length > 0
                         && search.results.length > 0 && !search.reranked
                         && !search.rerankerLoading
                highlighted: true
                onClicked: search.rerank()
                ToolTip.visible: hovered
                ToolTip.text: "Rescore these results with a cross-encoder"
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Label {
            opacity: 0.7
            text: {
                if (!search.ready || query.text.trim().length === 0)
                    return search.status
                if (search.results.length === 0)
                    return "No matches. Keyword search needs a word in common; try Hybrid or Semantic."
                const what = search.reranked ? "reranked by cross-encoder"
                           : search.mode === HelpSearch.Keyword ? "keyword (BM25)"
                           : search.mode === HelpSearch.Semantic ? "semantic" : "hybrid"
                return search.results.length + " results, " + what + ", in "
                       + search.elapsedMs.toFixed(1) + " ms"
            }
        }

        // Column header
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 8
            Layout.rightMargin: 8
            spacing: 12
            Label { text: "#"; Layout.preferredWidth: 28; Layout.minimumWidth: 28; Layout.maximumWidth: 28; font.bold: true }
            Label { text: "Article"; Layout.fillWidth: true; font.bold: true }
            Label { text: "Category"; Layout.preferredWidth: 130; Layout.minimumWidth: 130; Layout.maximumWidth: 130; font.bold: true }
            Label { text: "Score"; Layout.preferredWidth: 150; Layout.minimumWidth: 150; Layout.maximumWidth: 150; font.bold: true }
            Label { text: "Moved"; Layout.preferredWidth: 56; Layout.minimumWidth: 56; Layout.maximumWidth: 56; font.bold: true; visible: search.reranked }
        }

        ListView {
            id: list
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: search.results
            ScrollBar.vertical: ScrollBar {}

            delegate: Rectangle {
                required property var modelData
                required property int index
                width: ListView.view.width
                height: row.implicitHeight + 16
                color: index % 2 === 0 ? window.palette.base : window.palette.alternateBase

                RowLayout {
                    id: row
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 12

                    Label {
                        text: modelData.rank
                        Layout.preferredWidth: 28; Layout.minimumWidth: 28; Layout.maximumWidth: 28
                        Layout.alignment: Qt.AlignTop
                        opacity: 0.6
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        spacing: 2
                        Label {
                            text: modelData.title
                            font.bold: true
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Label {
                            text: modelData.body
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                            opacity: 0.75
                        }
                    }

                    Label {
                        text: modelData.category
                        Layout.preferredWidth: 130; Layout.minimumWidth: 130; Layout.maximumWidth: 130
                        Layout.alignment: Qt.AlignTop
                        opacity: 0.75
                    }

                    ColumnLayout {
                        Layout.preferredWidth: 150; Layout.minimumWidth: 150; Layout.maximumWidth: 150
                        Layout.alignment: Qt.AlignTop
                        spacing: 4
                        visible: modelData.score !== undefined
                        Label {
                            text: modelData.score !== undefined ? modelData.score.toFixed(4) : ""
                            font.family: "monospace"
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 4
                            radius: 2
                            color: window.palette.mid
                            Rectangle {
                                height: parent.height
                                radius: 2
                                color: window.palette.highlight
                                width: modelData.score !== undefined
                                       ? parent.width * window.barFraction(modelData.score) : 0
                            }
                        }
                    }

                    Label {
                        visible: search.reranked
                        Layout.preferredWidth: 56; Layout.minimumWidth: 56; Layout.maximumWidth: 56
                        Layout.alignment: Qt.AlignTop
                        font.family: "monospace"
                        readonly property int delta: modelData.previousRank - modelData.rank
                        text: delta > 0 ? "▲ " + delta : delta < 0 ? "▼ " + (-delta) : "–"
                        color: delta > 0 ? "#2e7d32" : delta < 0 ? "#c62828" : window.palette.text
                    }
                }
            }
        }
    }
}
