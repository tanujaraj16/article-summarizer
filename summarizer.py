from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

# Download required nltk data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary_sentences)

# Sample text
text = """
Artificial Intelligence (AI) is transforming industries across the globe.
From healthcare to finance, AI-driven solutions are improving efficiency,
reducing costs, and enabling smarter decision-making. In healthcare, AI
helps in predicting diseases, assisting doctors with diagnosis, and
personalizing patient care. In finance, AI algorithms are used to detect
fraudulent transactions and manage risks. Despite its benefits, AI also
raises ethical concerns such as job displacement, bias in decision-making,
and data privacy issues. As AI continues to evolve, it is important to
strike a balance between innovation and responsible usage.
"""

print("\n--- Summary ---\n")
print(summarize_text(text, sentence_count=2))
