# %%
from tika import parser

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter



epub = '../data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub'
parsed = parser.from_file(str(epub))
text = parsed["content"]

splitter = RecursiveCharacterTextSplitter(
    # length_function=lambda x: len(self.tts_model.tokenizer.encode(x, lang="en")),
    chunk_size=400,
    chunk_overlap=0,
    keep_separator=True,
    strip_whitespace=True,
    separators=[
        "\n\n", "\n", "\xa0", '<div>', '<p>', '<br>', "\r", ".",  "!", "?", 
        
        '"', "'", "‘", "’", "“", "”", "„", "‟",  
        "(", ")", "[", "]", "{", "}", 
        "…", ":", ";", "—",
        " ", '' # these ensure that there is always something to split by
        ],
)
texts = splitter.split_text(text)
ls = [splitter._length_function(x) for x in texts]
ls, max(ls)

# %%
"""
When splitting text for Language Models, aim for two properties:

 - Limit tokens to a maximum size (e.g., 400)
 - Use natural boundaries for splits (e.g. ".")

Many splitters don't enforce a token size limit, causing errors like "device assert" or "out of memory." Others focus on character length rather than token length. To address these issues:

- Use RecursiveCharacterTextSplitter from the langchain library
- Set the last separator to an empty string '' to ensure there is always a splitting point, thus maintaining token limits
- Utilize the encoder in the length function

"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo") # FIXME: use the right encoder for your model!

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    length_function=lambda x: len(encoder.encode(x)),
    chunk_size=100,
    chunk_overlap=0,
    keep_separator=True,
    strip_whitespace=False,
    add_start_index=True,
    separators=[
        # Sperators in order of priority
        "\n\n", "\n", "\xa0", '<div>', '<p>', '<br>', "\r", ".",  "!", "?",
        '"', "'", "‘", "’", "“", "”", "„", "‟",
        "(", ")", "[", "]", "{", "}",
        "…", ":", ";", "—",
        " ", ''  # These last two tokens ensure there is always a point to split by. Therefore keeping chunks<=limit
    ],
)

# Split the text (fro  `Meditations: The Annotated Edition. Book 2` Translator: Robin Waterfield Date: 2021)
text = """At the start of the day tell yourself: I shall meet people who are officious, ungrateful, abusive, treacherous, malicious, and selfish. In every case, they’ve got like this because of their ignorance of good and bad. But I have seen goodness and badness for what they are, and I know that what is good is what is morally right, and what is bad is what is morally wrong;and I’ve seen the true nature of the wrongdoer himself and know that he’s related to me—not in the sense that we share blood and seed, but by virtue of the fact that we both partake of the same intelligence, and so of a portion of the divine. None of them can harm me, anyway, because none of them can infect me with immorality,nor can I become angry with someone who’s related to me, or hate him, because we were born to work together, like feet or hands or eyelids, like the rows of upper and lower teeth. To work against each other is therefore unnatural—and anger and rejection count as “working against.”"""
texts = splitter.split_text(text)
texts

# %%
