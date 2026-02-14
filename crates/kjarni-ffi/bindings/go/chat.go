package kjarni

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type ChatMode int32

const (
	ChatDefault   ChatMode = 0
	ChatCreative  ChatMode = 1
	ChatReasoning ChatMode = 2
)

type GenerationConfig struct {
	Temperature     float32
	TopK            int32
	TopP            float32
	MinP            float32
	RepetitionPenalty float32
	MaxNewTokens    int32
	DoSample        int32
}

func DefaultGenerationConfig() GenerationConfig {
	return GenerationConfig{
		Temperature:     -1.0,
		TopK:            -1,
		TopP:            -1.0,
		MinP:            -1.0,
		RepetitionPenalty: -1.0,
		MaxNewTokens:    -1,
		DoSample:        -1,
	}
}

func GreedyGenerationConfig() GenerationConfig {
	return GenerationConfig{
		Temperature:     0.0,
		TopK:            -1,
		TopP:            -1.0,
		MinP:            -1.0,
		RepetitionPenalty: -1.0,
		MaxNewTokens:    -1,
		DoSample:        0,
	}
}

func CreativeGenerationConfig() GenerationConfig {
	return GenerationConfig{
		Temperature:     0.9,
		TopK:            50,
		TopP:            0.95,
		MinP:            -1.0,
		RepetitionPenalty: -1.0,
		MaxNewTokens:    -1,
		DoSample:        1,
	}
}

//  struct layouts

type ffiChatConfig struct {
	Device       int32
	_            int32 // padding
	CacheDir     uintptr
	ModelName    uintptr
	ModelPath    uintptr
	SystemPrompt uintptr
	Mode         int32
	Quiet        int32
}

type ffiGenerationConfig struct {
	Temperature     float32
	TopK            int32
	TopP            float32
	MinP            float32
	RepetitionPenalty float32
	MaxNewTokens    int32
	DoSample        int32
}

type ChatOption func(*chatOptions)

type chatOptions struct {
	quiet        bool
	device       string
	systemPrompt string
	mode         ChatMode
}

func WithSystemPrompt(prompt string) ChatOption {
	return func(o *chatOptions) {
		o.systemPrompt = prompt
	}
}

func WithChatMode(mode ChatMode) ChatOption {
	return func(o *chatOptions) {
		o.mode = mode
	}
}

type Chat struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

func NewChat(model string, opts ...interface{}) (*Chat, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	co := chatOptions{device: "cpu"}
	for _, opt := range opts {
		switch o := opt.(type) {
		case Option:
			// standard options
			var stdOpts options
			o(&stdOpts)
			if stdOpts.quiet {
				co.quiet = true
			}
			if stdOpts.device != "" {
				co.device = stdOpts.device
			}
		case ChatOption:
			o(&co)
		}
	}

	modelStr, keepModel := cString(model)
	defer keepModel()

	var config ffiChatConfig
	config.Device = deviceCode(co.device)
	config.ModelName = modelStr
	config.Mode = int32(co.mode)
	config.Quiet = boolToInt(co.quiet)

	if co.systemPrompt != "" {
		sp, keepSP := cString(co.systemPrompt)
		defer keepSP()
		config.SystemPrompt = sp
	}

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_chatNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Chat{handle: handle}, nil
}

// sends a message and returns the full response, blocking
func (c *Chat) Send(message string) (string, error) {
	return c.SendWithConfig(message, nil)
}

//  sends a message with custom generation config
func (c *Chat) SendWithConfig(message string, config *GenerationConfig) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return "", errors.New("chat is closed")
	}

	msgPtr, keepMsg := cString(message)
	defer keepMsg()

	var configPtr uintptr
	var ffiConfig ffiGenerationConfig
	if config != nil {
		ffiConfig = toFFIGenConfig(*config)
		configPtr = uintptr(unsafe.Pointer(&ffiConfig))
	}

	var resultPtr uintptr
	r1, _, _ := purego.SyscallN(
		_chatSendSym,
		c.handle,
		msgPtr,
		configPtr,
		uintptr(unsafe.Pointer(&resultPtr)),
	)

	code := int32(r1)
	if code != 0 {
		return "", lastError(code)
	}

	if resultPtr == 0 {
		return "", nil
	}
	result := goString(resultPtr)
	// Free the string allocated by Rust
	purego.SyscallN(_stringFreeSym, resultPtr)
	return result, nil
}

// Stream sends a message and calls onToken for each generated token.
// Return false from onToken to stop generation.
func (c *Chat) Stream(message string, onToken func(token string) bool) error {
	return c.StreamWithConfig(message, nil, onToken)
}

// streams with custom generation config.
func (c *Chat) StreamWithConfig(message string, config *GenerationConfig, onToken func(token string) bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return errors.New("chat is closed")
	}

	msgPtr, keepMsg := cString(message)
	defer keepMsg()

	var configPtr uintptr
	var ffiConfig ffiGenerationConfig
	if config != nil {
		ffiConfig = toFFIGenConfig(*config)
		configPtr = uintptr(unsafe.Pointer(&ffiConfig))
	}

	// callback
	cb := purego.NewCallback(func(textPtr uintptr, userData uintptr) uintptr {
		text := goString(textPtr)
		if onToken(text) {
			return 1 // continue
		}
		return 0 // stop
	})

	r1, _, _ := purego.SyscallN(
		_chatStreamSym,
		c.handle,
		msgPtr,
		configPtr,
		cb,
		0, // userData
		0, // cancelToken
	)

	code := int32(r1)
	if code != 0 {
		return lastError(code)
	}
	return nil
}

// returns model name
func (c *Chat) ModelName() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ""
	}

	needed, _, _ := purego.SyscallN(_chatModelNameSym, c.handle, 0, 0)
	if needed == 0 {
		return ""
	}

	buf := make([]byte, needed+1)
	purego.SyscallN(_chatModelNameSym, c.handle, uintptr(unsafe.Pointer(&buf[0])), needed+1)
	return string(buf[:needed])
}

// returns the model's context window size
func (c *Chat) ContextSize() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0
	}
	r, _, _ := purego.SyscallN(_chatContextSizeSym, c.handle)
	return int(r)
}

// create a stateful conversation with history management
func (c *Chat) Conversation() (*ChatConversation, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, errors.New("chat is closed")
	}

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_chatConversationNewSym,
		c.handle,
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &ChatConversation{handle: handle, chat: c}, nil
}

//releases the Chat resources
func (c *Chat) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	_chatFree(c.handle)
	c.handle = 0
	return nil
}

// ChatConversation maintains chat history
// parent Chat must outlive this object
type ChatConversation struct {
	handle uintptr
	chat   *Chat // prevent parent GC
	mu     sync.Mutex
	closed bool
}

// Send sends a message and returns a response
// Both user message and response are added to history
func (cc *ChatConversation) Send(message string) (string, error) {
	return cc.SendWithConfig(message, nil)
}

// send with custom generation config.
func (cc *ChatConversation) SendWithConfig(message string, config *GenerationConfig) (string, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.closed {
		return "", errors.New("conversation is closed")
	}

	msgPtr, keepMsg := cString(message)
	defer keepMsg()

	var configPtr uintptr
	var ffiConfig ffiGenerationConfig
	if config != nil {
		ffiConfig = toFFIGenConfig(*config)
		configPtr = uintptr(unsafe.Pointer(&ffiConfig))
	}

	var resultPtr uintptr
	r1, _, _ := purego.SyscallN(
		_chatConversationSendSym,
		cc.handle,
		msgPtr,
		configPtr,
		uintptr(unsafe.Pointer(&resultPtr)),
	)

	code := int32(r1)
	if code != 0 {
		return "", lastError(code)
	}

	if resultPtr == 0 {
		return "", nil
	}
	result := goString(resultPtr)
	purego.SyscallN(_stringFreeSym, resultPtr)
	return result, nil
}

// streams a response. Message and full response are added to history
func (cc *ChatConversation) Stream(message string, onToken func(token string) bool) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.closed {
		return errors.New("conversation is closed")
	}

	msgPtr, keepMsg := cString(message)
	defer keepMsg()

	cb := purego.NewCallback(func(textPtr uintptr, userData uintptr) uintptr {
		text := goString(textPtr)
		if onToken(text) {
			return 1
		}
		return 0
	})

	r1, _, _ := purego.SyscallN(
		_chatConversationStreamSym,
		cc.handle,
		msgPtr,
		0, // default genConfig
		cb,
		0, // userData
		0, // cancelToken
	)

	code := int32(r1)
	if code != 0 {
		return lastError(code)
	}
	return nil
}

// returns the number of messages in history
func (cc *ChatConversation) Length() int {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if cc.closed {
		return 0
	}
	r, _, _ := purego.SyscallN(_chatConversationLenSym, cc.handle)
	return int(r)
}

// clear the conversation history
func (cc *ChatConversation) Clear(keepSystem bool) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if cc.closed {
		return
	}
	ks := uintptr(0)
	if keepSystem {
		ks = 1
	}
	purego.SyscallN(_chatConversationClearSym, cc.handle, ks)
}

// release the conversation resources
func (cc *ChatConversation) Close() error {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.closed {
		return nil
	}
	cc.closed = true
	_chatConversationFree(cc.handle)
	cc.handle = 0
	return nil
}

func toFFIGenConfig(config GenerationConfig) ffiGenerationConfig {
	return ffiGenerationConfig{
		Temperature:     config.Temperature,
		TopK:            config.TopK,
		TopP:            config.TopP,
		MinP:            config.MinP,
		RepetitionPenalty: config.RepetitionPenalty,
		MaxNewTokens:    config.MaxNewTokens,
		DoSample:        config.DoSample,
	}
}