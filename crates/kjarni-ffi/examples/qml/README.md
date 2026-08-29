# Kjarni in Qt Quick

Local inference in a QML application. The model loads in the background, results
arrive as signals, and `busy` drives the spinner — QML never learns it is talking to
a transformer.

```qml
KjarniEmbedder {
    id: embedder
    model: "minilm-l6-v2"

    Component.onCompleted: load()
    onReady: status.text = "Model ready. Nothing leaves this machine."
    onRanked: (results) => { for (const r of results) hits.append(r) }
}

Button {
    enabled: embedder.ready && !embedder.busy
    onClicked: embedder.rank(query.text, documents)
}
```

Chat streams token by token:

```qml
KjarniChat {
    id: chat
    model: "llama3.2-1b-instruct"
    onToken: (text) => reply.text += text
}
```

## Why the threading matters

Inference is compute-bound and takes anywhere from milliseconds to several seconds.
Running it on Qt's GUI thread freezes the window for that entire time — the single
most common way a desktop AI feature ends up feeling broken.

`KjarniModel.h` dispatches every call through `QtConcurrent::run` and marshals the
result back with `QFutureWatcher`, so the event loop keeps running. The streaming
callback fires on a pool thread, so it queues each token onto the object's own thread
with `QMetaObject::invokeMethod(..., Qt::QueuedConnection)` before emitting. Touching
QML from a pool thread is a data race, not a slow path.

The only thing you have to respect: **one instance serves one request at a time.**
Generation is not re-entrant. The `busy` and `generating` properties exist so the UI
can reflect that rather than queueing calls that will be dropped.

## Building

Needs Qt 6.5 or newer, for `QML_ELEMENT` auto-registration and `loadFromModule`.

```bash
cmake -B build \
    -DCMAKE_PREFIX_PATH=/path/to/Qt/6.8.0/gcc_64 \
    -DKJARNI_LIB_DIR=/path/to/target/release
cmake --build build
./build/kjarni-qml
```

`KjarniModel.h` is listed under `SOURCES` in `qt_add_qml_module` rather than as a
plain header, so AUTOMOC processes its `Q_OBJECT` classes and the `QML_ELEMENT`
registrations reach the type system.

## What is verified, and what is not

**Not compiled.** Qt was not available on the machine this was written on, so the
Qt-specific code — signals, properties, `QFutureWatcher` wiring, the QML file — has
not been built or run. Treat it as a starting point rather than tested code.

What *has* been checked is everything underneath it: `kjarni.hpp` compiles clean
under `-Wall -Wextra`, runs clean under AddressSanitizer and UndefinedBehaviorSanitizer
with leak detection, and the exact call patterns this wrapper uses — `std::optional`
storage, moving out of the factories, a stream callback capturing by reference — are
covered by compile-time assertions and by the console example in
[`../cpp`](../cpp), which does run.

If the Qt build fights you, the likely places are the AUTOMOC setup and the Qt
version floor, not the Kjarni calls.

## Requirements

- Qt 6.5+ (Quick, Concurrent)
- C++23, for `std::expected` in `kjarni.hpp`
- `libkjarni_ffi` for your platform

## Where this fits

Qt applications are frequently the case where a cloud API is not an option: desktop
software shipped to customers, industrial and medical systems, field tools that run
without a network. A single shared library with a C header is a shape that Qt
projects already consume comfortably, and unlike .NET there is no established local
inference story to compete with.
