"""Keep stdlib callback keyword compatibility during lint-oriented cleanups."""

import tempfile
import unittest
from pathlib import Path

from work.check_docs_links import PageLinks
from work.docs_access import ArticleLinks
from wtf.server import State, make_handler


class CallbackContractTests(unittest.TestCase):
    """Regression coverage for callback signatures inherited from the stdlib."""

    def test_html_callbacks_accept_stdlib_attribute_keyword(self):
        """Both link parsers accept attrs just like HTMLParser.handle_starttag."""
        page = PageLinks()
        page.handle_starttag(tag="a", attrs=[("href", "guide/")])
        self.assertIn("guide/", page.links)
        article = ArticleLinks()
        article.handle_starttag(tag="article", attrs=[])
        article.handle_starttag(tag="a", attrs=[("href", "guide/")])
        article.handle_data("Guide")
        article.handle_endtag("a")
        self.assertEqual(article.links, [("guide/", "Guide")])

    def test_quiet_handler_accepts_stdlib_format_keyword(self):
        """Quiet logging remains compatible with the inherited format keyword."""
        with tempfile.TemporaryDirectory() as folder:
            handler = make_handler(State(Path(folder)))
            self.assertIsNone(handler.log_message(None, format="hidden"))
            self.assertIsNone(handler.log_message(None, "hidden %s", "value"))
