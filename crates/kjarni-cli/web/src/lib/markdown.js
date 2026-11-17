import { Marked } from 'marked';
import DOMPurify from 'dompurify';

const marked = new Marked({ gfm: true, breaks: true });

// Links in replies open in a new tab and tell the target nothing.
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

/** Model text as safe HTML. Replies can repeat what a fetched web page or a
 * file said, so nothing in them may run in this page, which can approve the
 * agent's actions. */
export function renderMarkdown(text) {
  const html = marked.parse(text ?? '');
  return DOMPurify.sanitize(html, {
    FORBID_TAGS: ['style', 'form', 'input', 'button', 'textarea', 'select', 'iframe', 'img'],
    FORBID_ATTR: ['style'],
  });
}
