package kjarni

type options struct {
	quiet    bool
	device   string // "cpu" || "gpu"
	cacheDir string

	// Indexer
	extensions    []string
	exclude       []string
	chunkSize     int
	chunkOverlap  int
	notRecursive  bool
	includeHidden bool

	// Transcriber
	language  string
	translate bool

	// Translator
	from, to string

	// Summarizer
	minLength, maxLength int
}

// Option configures a classifier, embedder, or other kjarni component.
type Option func(*options)

// WithQuiet suppresses log output during model loading and inference.
func WithQuiet(quiet bool) Option {
	return func(o *options) {
		o.quiet = quiet
	}
}

// WithDevice sets the compute device. Supported values: "cpu", "gpu".
func WithDevice(device string) Option {
	return func(o *options) {
		o.device = device
	}
}

// WithCacheDir sets where models are downloaded to and loaded from.
func WithCacheDir(dir string) Option {
	return func(o *options) {
		o.cacheDir = dir
	}
}

// WithExtensions limits an Indexer to files with these extensions, e.g. "md", "txt".
func WithExtensions(extensions ...string) Option {
	return func(o *options) {
		o.extensions = extensions
	}
}

// WithExclude makes an Indexer skip names or `*` patterns, e.g. "node_modules", "*.min.js".
func WithExclude(patterns ...string) Option {
	return func(o *options) {
		o.exclude = patterns
	}
}

// WithChunking sets an Indexer's chunk size and the overlap between chunks, in characters.
func WithChunking(size, overlap int) Option {
	return func(o *options) {
		o.chunkSize = size
		o.chunkOverlap = overlap
	}
}

// WithRecursive sets whether an Indexer descends into subdirectories (the default).
func WithRecursive(recursive bool) Option {
	return func(o *options) {
		o.notRecursive = !recursive
	}
}

// WithHidden makes an Indexer include hidden files and directories.
func WithHidden(include bool) Option {
	return func(o *options) {
		o.includeHidden = include
	}
}

// WithLanguage sets the spoken language for a Transcriber as an ISO code such as
// "en" or "is". Without it the language is detected from the audio.
func WithLanguage(language string) Option {
	return func(o *options) {
		o.language = language
	}
}

// WithTranslate makes a Transcriber translate the speech into English.
func WithTranslate(translate bool) Option {
	return func(o *options) {
		o.translate = translate
	}
}

// WithLanguages sets a Translator's default source and target languages.
func WithLanguages(from, to string) Option {
	return func(o *options) {
		o.from = from
		o.to = to
	}
}

// WithSummaryLength bounds a Summarizer's output in tokens; 0 keeps the model's default.
func WithSummaryLength(min, max int) Option {
	return func(o *options) {
		o.minLength = min
		o.maxLength = max
	}
}

func applyOptions(opts []Option) options {
	o := options{
		device: "cpu",
	}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

func deviceCode(device string) int32 {
	if device == "gpu" {
		return 1
	}
	return 0
}

func boolToInt(b bool) int32 {
	if b {
		return 1
	}
	return 0
}