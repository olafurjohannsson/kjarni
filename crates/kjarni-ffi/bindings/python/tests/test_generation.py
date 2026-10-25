import unittest

from _support import heavy
from kjarni import Chat, Summarizer, Translator, KjarniException
from kjarni._ffi import KjarniError

ARTICLE = (
    "The city council voted on Tuesday to approve a new public library in the "
    "downtown district. The library will include a children's wing, a maker space "
    "with 3D printers, and a cafe. Construction is expected to begin next spring and "
    "take about eighteen months. Council members said the project would be funded "
    "through a combination of municipal bonds and private donations. Several "
    "residents spoke in favour of the plan at the meeting, citing the lack of quiet "
    "study spaces in the area. A small group opposed it, arguing the money would be "
    "better spent on road repairs. The mayor said the library would become a "
    "cornerstone of the community and help revitalise the downtown area, which has "
    "seen several businesses close over the past few years."
)


@heavy
class SummarizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summarizer = Summarizer(quiet=True)

    def test_summarizes_an_article(self):
        summary = self.summarizer.summarize(ARTICLE)
        self.assertTrue(summary.strip())
        self.assertLess(len(summary), len(ARTICLE))

    def test_unknown_model_fails(self):
        with self.assertRaises(KjarniException) as e:
            Summarizer(model="no-such-model", quiet=True)
        self.assertEqual(e.exception.code, KjarniError.MODEL_NOT_FOUND)


@heavy
class TranslatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = Translator(source="en", target="de", quiet=True)

    def test_default_languages(self):
        self.assertIn("haus", self.translator.translate("The house is small.").lower())

    def test_languages_per_call(self):
        out = self.translator.translate("The house is small.", source="English", target="German")
        self.assertIn("haus", out.lower())

    def test_no_languages_is_invalid_config(self):
        bare = Translator(quiet=True)
        with self.assertRaises(KjarniException) as e:
            bare.translate("hello")
        self.assertEqual(e.exception.code, KjarniError.INVALID_CONFIG)


@heavy
class ChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chat = Chat("qwen2.5-0.5b-instruct", quiet=True)

    def test_send(self):
        reply = self.chat.send("What is the capital of France? Answer in one word.",
                               do_sample=False, max_new_tokens=16)
        self.assertIn("paris", reply.lower())

    def test_stream_matches_send(self):
        prompt = "Name a primary colour. One word."
        sent = self.chat.send(prompt, do_sample=False, max_new_tokens=8)
        tokens = []
        streamed = self.chat.stream(prompt, tokens.append, do_sample=False, max_new_tokens=8)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(streamed, "".join(tokens))
        self.assertEqual(streamed.strip(), sent.strip())

    def test_stream_stops_when_callback_returns_false(self):
        tokens = []
        self.chat.stream("Count from one to fifty.", lambda t: tokens.append(t) or len(tokens) < 3,
                         do_sample=False, max_new_tokens=64)
        self.assertEqual(len(tokens), 3)

    def test_callback_exception_propagates(self):
        def boom(_):
            raise RuntimeError("from the callback")
        with self.assertRaisesRegex(RuntimeError, "from the callback"):
            self.chat.stream("Hello", boom, max_new_tokens=8)

    def test_history(self):
        history = [("user", "My name is Anna."), ("assistant", "Nice to meet you, Anna.")]
        reply = self.chat.send_with_history(history, "What is my name?", do_sample=False, max_new_tokens=16)
        self.assertIn("anna", reply.lower())
        with self.assertRaises(ValueError):
            self.chat.send_with_history([("narrator", "x")], "hi")

    def test_conversation_keeps_history(self):
        convo = self.chat.conversation()
        convo.send("Remember this: the secret word is banana.", do_sample=False, max_new_tokens=24)
        self.assertGreaterEqual(len(convo), 2)
        reply = convo.send("What is the secret word?", do_sample=False, max_new_tokens=24)
        self.assertIn("banana", reply.lower())
        # The stateless call has no history, so the word is not there to recall.
        self.assertNotIn("banana", self.chat.send("What is the secret word?", do_sample=False,
                                                  max_new_tokens=24).lower())
        convo.clear()
        self.assertLessEqual(len(convo), 1)

    def test_metadata(self):
        self.assertTrue(self.chat.model_name)
        self.assertGreater(self.chat.context_size, 0)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            Chat("qwen2.5-0.5b-instruct", mode="sleepy")


if __name__ == "__main__":
    unittest.main()
