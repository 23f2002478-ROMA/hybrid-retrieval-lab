"""
Virtual Lab Experiment 13: Hybrid Keyword and Semantic Retrieval
Combines BM25 lexical retrieval with embedding-based (LSA) semantic
retrieval into a single tunable hybrid ranking.
Sections: Theory, Simulation, Quiz, Report Generation.
"""

import os
import re
import math
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# 1. EXPERIMENT CONFIGURATION AND EDUCATIONAL CONTENT

EXPERIMENT_CONFIG = {
    "title": "Hybrid Keyword and Semantic Retrieval",
    "objectives": [
        "Understand how BM25 scores documents using term frequency and inverse document frequency.",
        "Understand how embedding-based semantic search captures meaning beyond exact keyword overlap.",
        "Combine lexical (BM25) and semantic scores into a single hybrid ranking using a tunable weight alpha.",
        "Analyze how alpha, k1 and b change document rankings for different queries.",
        "Evaluate whether hybrid retrieval improves result relevance compared to either method alone."
    ]
}

THEORY_CONTENT = {
    "background": """
### Lexical retrieval: BM25
BM25 ranks a document D for a query Q by summing, over each query term t,
an inverse-document-frequency weight times a saturated term-frequency term:

score(D, Q) = sum over t in Q of  IDF(t) * (f(t,D) * (k1+1)) / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))

- f(t,D) is how many times term t appears in document D.
- IDF(t) is high for rare terms and low for common terms.
- k1 controls how quickly extra occurrences of a term stop adding score (term-frequency saturation).
- b controls how much longer documents are penalized (length normalization).

BM25 only ever "sees" the exact tokens in the query, so it misses documents
that are relevant but worded differently (e.g. query "vector representation
of words" vs a document about "word embeddings").

### Semantic retrieval: embeddings
Semantic search represents both the query and every document as a dense
vector, then ranks documents by cosine similarity of their vector to the
query vector. Vectors that are close in this space tend to be close in
meaning, even with little or no word overlap.

This lab builds its embeddings with Latent Semantic Analysis (LSA): a
TF-IDF matrix of the corpus is projected onto its top singular directions
(TruncatedSVD). This keeps the simulator small and self-contained while
still producing genuine semantic vectors, in the same spirit as larger
neural embedding models.

### Hybrid combination
Both score types are first min-max normalized to [0, 1] so neither scale
dominates, then blended with a single weight alpha:

hybrid_score(D) = alpha * semantic_score(D) + (1 - alpha) * bm25_score(D)

- alpha = 0 reduces to pure BM25.
- alpha = 1 reduces to pure semantic search.
- Intermediate alpha lets exact keyword matches and semantically related
  documents both contribute to the final ranking.
    """,
    "procedure": [
        "Step 1: Read the theory below and note the BM25 and hybrid-score formulas.",
        "Step 2: Go to the Simulation section and enter or pick a query.",
        "Step 3: Set alpha, k1 and b, and choose how many top results (k) to inspect.",
        "Step 4: Compare the BM25-only, semantic-only and hybrid rankings and their overlap.",
        "Step 5: Click 'Record Current Trial' to log the configuration and outcome.",
        "Step 6: Repeat for at least 3-4 different queries and alpha values.",
        "Step 7: Complete the assessment Quiz.",
        "Step 8: Open Report Generation, fill in your details, and download the PDF report."
    ],
    "key_terms": {
        "BM25": "Lexical ranking function based on term frequency and inverse document frequency.",
        "TF / IDF": "Term frequency: how often a term occurs in a document. Inverse document frequency: how rare a term is across the corpus.",
        "k1": "BM25 parameter controlling how fast repeated term occurrences saturate.",
        "b": "BM25 parameter controlling how strongly document length is penalized.",
        "Embedding": "A dense numeric vector representation of text that captures meaning.",
        "LSA": "Latent Semantic Analysis: embeddings derived from an SVD of the TF-IDF matrix.",
        "Cosine similarity": "Similarity between two vectors based on the angle between them.",
        "Alpha": "Hybrid weight that blends the semantic score and the BM25 score.",
        "Overlap@k": "Fraction of the top-k hybrid results that are also in the top-k BM25-only results."
    }
}

CORPUS = [
    ("D1", "Convolutional neural networks are the dominant architecture for modern computer vision systems, "
           "using stacked layers of learnable filters to detect edges, textures, and increasingly abstract "
           "visual patterns as the network gets deeper. A typical pipeline resizes and normalizes input images, "
           "passes them through convolutional and pooling layers to build a hierarchical feature representation, "
           "and finally uses fully connected layers to output class probabilities for tasks such as image "
           "classification, object detection, and semantic segmentation. Techniques like data augmentation, "
           "batch normalization, and transfer learning from large pretrained models such as ResNet or "
           "EfficientNet have made it practical to reach high accuracy even with limited labeled training data."),

    ("D2", "Classical machine learning algorithms such as support vector machines, decision trees, and random "
           "forests remain widely used for structured, tabular data where deep learning offers little advantage. "
           "A support vector machine finds the hyperplane that best separates classes with the maximum margin, "
           "optionally using a kernel trick to handle non-linearly separable data. Decision trees recursively "
           "split the feature space based on the most informative feature at each step, which makes them easy "
           "to interpret but prone to overfitting; random forests address this by training many decision trees "
           "on bootstrapped samples and averaging their predictions, trading some interpretability for a large "
           "gain in generalization performance."),

    ("D3", "BM25 is a probabilistic ranking function used by traditional search engines to score how relevant a "
           "document is to a query. For every query term, it combines an inverse document frequency weight, "
           "which rewards rare and informative terms, with a saturating term frequency component controlled by "
           "the parameter k1, so that a term appearing many times in a document does not dominate the score "
           "indefinitely. A second parameter, b, controls how strongly a document's length relative to the "
           "average document length in the collection is penalized, since a long document may simply contain "
           "more term occurrences by chance rather than being more relevant. Despite predating modern neural "
           "retrieval methods, BM25 remains a strong and efficient baseline in production search systems."),

    ("D4", "Modern natural language processing relies heavily on dense vector representations of text, known "
           "as embeddings, that place semantically similar words or sentences close together in a continuous "
           "vector space even when they share no words in common. Earlier approaches such as word2vec and "
           "GloVe learned static word embeddings from large text corpora, while transformer based models like "
           "BERT and GPT produce contextual embeddings that change depending on the surrounding sentence, using "
           "a self attention mechanism to weigh the relevance of every other token when encoding a given "
           "position. These embeddings power tasks ranging from sentiment analysis and named entity recognition "
           "to semantic search, where a query and a document can be matched by the similarity of their vectors "
           "instead of by exact keyword overlap."),

    ("D5", "Relational database systems rely on indexing structures, most commonly B-trees, to avoid scanning "
           "an entire table when answering a query that filters or sorts on an indexed column. A B-tree keeps "
           "its keys sorted and balanced so that lookups, insertions, and deletions all complete in logarithmic "
           "time relative to the number of rows, which is essential once a table grows to millions of records. "
           "Queries expressed in SQL frequently combine data from multiple tables using joins, and the "
           "database's query planner decides which indexes to use and in what order to execute joins based on "
           "estimated row counts and selectivity, aiming to minimize the total amount of disk and memory work "
           "needed to produce the result set."),

    ("D6", "Computer networks depend on a stack of protocols working together to move data reliably between "
           "machines. The Transmission Control Protocol, or TCP, breaks a stream of application data into "
           "segments, numbers them, and retransmits any that are lost or corrupted, guaranteeing that the "
           "receiving application sees an ordered, reliable byte stream even over an unreliable underlying "
           "network. Before two hosts can communicate by name, the Domain Name System translates a human "
           "readable hostname into a numeric IP address through a hierarchy of resolvers and authoritative "
           "servers. Firewalls sit at network boundaries and inspect this traffic, allowing or blocking packets "
           "based on rules defined over source and destination addresses, ports, and protocols, forming a first "
           "line of defense against unwanted or malicious connections."),

    ("D7", "An operating system kernel is responsible for sharing the limited physical resources of a computer, "
           "particularly the CPU and memory, among many competing processes. A scheduler decides which ready "
           "process gets to run on the CPU next, balancing goals such as overall throughput, fairness between "
           "processes, and responsiveness for interactive workloads, using strategies ranging from simple round "
           "robin scheduling to more elaborate priority based or multilevel feedback queues. Virtual memory "
           "extends this resource sharing to RAM by giving each process the illusion of its own large, "
           "contiguous address space, transparently mapping pages of that address space to physical memory "
           "frames and swapping less recently used pages out to disk when physical memory runs low."),

    ("D8", "Large scale computation and storage problems often exceed the capacity of a single machine, which "
           "motivates distributed systems that coordinate many independent computers to behave, from the "
           "outside, like one coherent service. The MapReduce programming model simplifies writing such large "
           "scale, parallel computations by asking the programmer to specify only a map function, which "
           "transforms input records independently, and a reduce function, which aggregates the intermediate "
           "results, while the underlying framework handles distributing the work across a cluster, restarting "
           "failed tasks, and moving data to where it is needed. In front of many distributed services, load "
           "balancers spread incoming requests across a pool of backend servers, monitoring their health and "
           "adjusting traffic so that no single server becomes a bottleneck or a single point of failure."),

    ("D9", "Modern secure communication depends on public key cryptography, in which every participant holds a "
           "mathematically related pair of keys: a public key that can be shared openly and a private key that "
           "must be kept secret. Data encrypted with a recipient's public key can only be decrypted with the "
           "matching private key, which allows two parties who have never met to establish a confidential "
           "channel over an insecure network, as is done during a TLS handshake when browsing a secure website. "
           "Cryptographic hash functions such as SHA-256 play a complementary role, taking an input of any size "
           "and producing a fixed length digest such that even a tiny change to the input produces a completely "
           "different output, which makes them useful for verifying data integrity and for storing password "
           "credentials without keeping the original password in plain text."),

    ("D10", "As software projects grow beyond a single developer working alone, teams rely on version control "
            "systems such as Git to track every change made to the source code over time, who made it, and why. "
            "Git stores the history of a project as a sequence of snapshots rather than a simple list of file "
            "differences, which makes operations such as branching to work on a new feature in isolation, and "
            "later merging that work back into the main line of development, both fast and safe even when many "
            "people are editing the same files concurrently. Combined with continuous integration pipelines "
            "that automatically build and test every proposed change, this workflow lets teams catch "
            "integration problems early and ship software with far greater confidence than manually "
            "coordinating changes over email or shared folders ever could.")
]

SIMULATION_CONFIG = {
    "preset_queries": [
        "vector representation of words",
        "how search engines rank documents",
        "algorithm for classification",
        "penalizing long documents in ranking",
        "combining rows from multiple tables",
        "translating domain names into ip addresses",
        "deciding which process runs next on the cpu",
        "splitting work across a cluster of machines",
        "secure communication using key pairs",
        "keep track of code changes over time"
    ],
    "default_query": "vector representation of words",
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_default": 0.5,
    "alpha_step": 0.05,
    "k1_min": 0.5,
    "k1_max": 3.0,
    "k1_default": 1.5,
    "k1_step": 0.1,
    "b_min": 0.0,
    "b_max": 1.0,
    "b_default": 0.75,
    "b_step": 0.05,
    "topk_options": [3, 5, 8]
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What does BM25 primarily rely on to score document relevance?",
        "options": [
            "A) Term frequency and inverse document frequency, adjusted for document length",
            "B) Random sampling of document words",
            "C) The document's file size in bytes",
            "D) The number of images in a document"
        ],
        "answer_index": 0,
        "explanation": "BM25 combines term frequency with an IDF weight and a length-normalization term."
    },
    {
        "id": 2,
        "question": "What does the parameter b control in the BM25 formula?",
        "options": [
            "A) Term-frequency saturation speed",
            "B) The strength of document length normalization",
            "C) The number of documents to retrieve",
            "D) The embedding dimension"
        ],
        "answer_index": 1,
        "explanation": "b scales how much a document's length relative to the average length affects its score."
    },
    {
        "id": 3,
        "question": "What does the parameter k1 control?",
        "options": [
            "A) How quickly extra occurrences of a query term stop adding to the score",
            "B) Length normalization",
            "C) The number of retrieved documents",
            "D) Vocabulary size"
        ],
        "answer_index": 0,
        "explanation": "k1 controls term-frequency saturation: higher k1 lets repeated terms keep contributing more score."
    },
    {
        "id": 4,
        "question": "Semantic retrieval methods primarily represent text as what?",
        "options": [
            "A) Exact keyword strings",
            "B) Dense numeric vectors that capture meaning",
            "C) Regular expressions",
            "D) A sorted alphabetical index"
        ],
        "answer_index": 1,
        "explanation": "Semantic methods embed text into a vector space where meaning, not exact words, determines closeness."
    },
    {
        "id": 5,
        "question": "In this lab, how is semantic similarity between a query and a document measured?",
        "options": [
            "A) Euclidean distance between raw text strings",
            "B) Cosine similarity between LSA vectors",
            "C) Levenshtein edit distance",
            "D) Hamming distance"
        ],
        "answer_index": 1,
        "explanation": "The simulator computes cosine similarity between the query's and each document's LSA vector."
    },
    {
        "id": 6,
        "question": "Why can pure BM25 miss a relevant document for a query like 'vector representation of words'?",
        "options": [
            "A) BM25 relies on exact/overlapping terms and misses documents worded differently but related in meaning",
            "B) BM25 always returns a score of zero",
            "C) BM25 ignores term frequency completely",
            "D) BM25 requires embeddings to run at all"
        ],
        "answer_index": 0,
        "explanation": "A document about 'word embeddings' may share few exact tokens with the query, so BM25 can rank it low even though it is relevant."
    },
    {
        "id": 7,
        "question": "What does the alpha weight control in hybrid retrieval?",
        "options": [
            "A) The number of documents in the corpus",
            "B) The blend between the semantic score and the BM25 score in the final ranking",
            "C) Document length normalization only",
            "D) Formatting of the PDF report only"
        ],
        "answer_index": 1,
        "explanation": "hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score, so alpha sets the blend."
    },
    {
        "id": 8,
        "question": "Why are BM25 and semantic scores normalized before being combined?",
        "options": [
            "A) They can have very different ranges, so normalizing keeps one from unfairly dominating the blend",
            "B) Streamlit requires all numbers to be between 0 and 1",
            "C) It removes the need for a query",
            "D) It converts the scores into probabilities of relevance"
        ],
        "answer_index": 0,
        "explanation": "BM25 and cosine-similarity scores live on different scales, so min-max normalization puts them on comparable footing."
    },
    {
        "id": 9,
        "question": "What does the Overlap@k metric shown in the simulator indicate?",
        "options": [
            "A) How many documents exist in the corpus",
            "B) How similar the BM25-only top-k ranking is to the hybrid top-k ranking",
            "C) The number of unique terms in the query",
            "D) The search response time"
        ],
        "answer_index": 1,
        "explanation": "Overlap@k is the fraction of the hybrid top-k results that also appear in the BM25-only top-k results."
    },
    {
        "id": 10,
        "question": "What is a key expected benefit of hybrid keyword plus semantic retrieval over either method alone?",
        "options": [
            "A) It always returns fewer results",
            "B) It combines exact-term precision with semantic recall for improved overall retrieval quality",
            "C) It removes the need for any ranking function",
            "D) It guarantees zero search latency"
        ],
        "answer_index": 1,
        "explanation": "Hybrid retrieval keeps BM25's precision on exact matches while adding semantic recall for related, differently worded documents."
    }
]


# 2. RETRIEVAL ENGINE (BM25 + LSA SEMANTIC + HYBRID BLEND)

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource
def build_index():
    doc_ids = [d[0] for d in CORPUS]
    doc_texts = [d[1] for d in CORPUS]
    doc_tokens = [tokenize(t) for t in doc_texts]
    doc_len = [len(t) for t in doc_tokens]
    avgdl = sum(doc_len) / len(doc_len)
    n_docs = len(CORPUS)

    df = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(doc_texts)
    n_comp = min(10, min(x.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    doc_vecs = svd.fit_transform(x)

    return {
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "doc_tokens": doc_tokens,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "n_docs": n_docs,
        "df": df,
        "vectorizer": vectorizer,
        "svd": svd,
        "doc_vecs": doc_vecs
    }


def idf(term, df, n_docs):
    n = df.get(term, 0)
    return math.log((n_docs - n + 0.5) / (n + 0.5) + 1)


def bm25_scores(query, k1, b, idx):
    q_terms = tokenize(query)
    scores = []
    for i, toks in enumerate(idx["doc_tokens"]):
        tf = Counter(toks)
        dl = idx["doc_len"][i]
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            num = f * (k1 + 1)
            den = f + k1 * (1 - b + b * dl / idx["avgdl"])
            s += idf(term, idx["df"], idx["n_docs"]) * (num / den)
        scores.append(s)
    return np.array(scores)


def semantic_scores(query, idx):
    q_vec = idx["svd"].transform(idx["vectorizer"].transform([query]))
    sims = cosine_similarity(q_vec, idx["doc_vecs"])[0]
    return sims


def minmax(a):
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def run_hybrid_search(query, alpha, k1, b, top_k, idx):
    bm = bm25_scores(query, k1, b, idx)
    sem = semantic_scores(query, idx)
    bm_n = minmax(bm)
    sem_n = minmax(sem)
    hyb = alpha * sem_n + (1 - alpha) * bm_n

    order = np.argsort(-hyb)
    bm_order = np.argsort(-bm)
    top_hyb = set(order[:top_k].tolist())
    top_bm = set(bm_order[:top_k].tolist())
    overlap = len(top_hyb & top_bm) / top_k

    rows = []
    for rank, i in enumerate(order[:top_k], start=1):
        rows.append({
            "Rank": rank,
            "Doc ID": idx["doc_ids"][i],
            "Snippet": idx["doc_texts"][i][:70] + ("..." if len(idx["doc_texts"][i]) > 70 else ""),
            "BM25 Score": round(float(bm_n[i]), 3),
            "Semantic Score": round(float(sem_n[i]), 3),
            "Hybrid Score": round(float(hyb[i]), 3)
        })
    results_df = pd.DataFrame(rows)

    return {
        "results_df": results_df,
        "top_bm25_doc": idx["doc_ids"][bm_order[0]],
        "top_semantic_doc": idx["doc_ids"][int(np.argmax(sem))],
        "top_hybrid_doc": idx["doc_ids"][order[0]],
        "overlap": overlap,
        "bm_n": bm_n,
        "sem_n": sem_n,
        "hyb": hyb,
        "order": order
    }


# 3. LAB REPORT PDF EXPORTER

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name, student_id, date_str, trials_df,
                         quiz_score, quiz_total, student_notes):
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)

    pdf.ln(12)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(14, int(190 / max(1, num_cols)))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:14], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:14], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "Hybrid retrieval was compared against BM25-only and semantic-only rankings across several "
        "queries, showing improved coverage of semantically related but lexically different documents."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


# 4. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT

def render_theory_section():
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- Goal {i+1}: {obj}")

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology & Variable Reference"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Variable", "Definition & Role"]
        )
        st.table(var_df)

    st.divider()
    with st.expander("Document Corpus Used in the Simulation"):
        corpus_df = pd.DataFrame(CORPUS, columns=["Doc ID", "Text"])
        st.table(corpus_df)


def render_simulation_section():
    st.header("Interactive Simulation Sandbox")
    st.info("Pick or type a query, tune alpha/k1/b, and compare BM25-only, semantic-only and hybrid rankings.")

    idx = build_index()
    cfg = SIMULATION_CONFIG

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = cfg["default_query"]

    col_q1, col_q2 = st.columns([2, 1])
    with col_q2:
        preset = st.selectbox("Preset queries", options=cfg["preset_queries"])
        if st.button("Use preset query", use_container_width=True):
            st.session_state["query_text"] = preset
    with col_q1:
        query = st.text_input("Query", key="query_text")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha = st.slider("Alpha (semantic weight)", cfg["alpha_min"], cfg["alpha_max"],
                           cfg["alpha_default"], cfg["alpha_step"])
    with col2:
        k1 = st.slider("k1 (term-frequency saturation)", cfg["k1_min"], cfg["k1_max"],
                        cfg["k1_default"], cfg["k1_step"])
    with col3:
        b = st.slider("b (length normalization)", cfg["b_min"], cfg["b_max"],
                       cfg["b_default"], cfg["b_step"])
    with col4:
        top_k = st.selectbox("Top-k", options=cfg["topk_options"], index=1)

    if not query.strip():
        st.warning("Enter a query to run the search.")
        return

    res = run_hybrid_search(query, alpha, k1, b, top_k, idx)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top BM25-only doc", res["top_bm25_doc"])
    with m2:
        st.metric("Top semantic-only doc", res["top_semantic_doc"])
    with m3:
        st.metric("Top hybrid doc", res["top_hybrid_doc"])
    with m4:
        st.metric(f"Overlap@{top_k}", f"{res['overlap']*100:.0f}%")

    st.subheader("Score Comparison for Top Results")
    show_ids = [idx["doc_ids"][i] for i in res["order"][:top_k]]
    show_idx = [idx["doc_ids"].index(d) for d in show_ids]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="BM25", x=show_ids, y=[res["bm_n"][i] for i in show_idx]))
    fig.add_trace(go.Bar(name="Semantic", x=show_ids, y=[res["sem_n"][i] for i in show_idx]))
    fig.add_trace(go.Bar(name="Hybrid", x=show_ids, y=[res["hyb"][i] for i in show_idx]))
    fig.update_layout(
        barmode="group",
        title="Normalized BM25 vs Semantic vs Hybrid Scores",
        xaxis_title="Document",
        yaxis_title="Normalized Score",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranked Hybrid Results")
    st.dataframe(res["results_df"], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current query and configuration into your session trial table:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": query,
                "Alpha": alpha,
                "k1": k1,
                "b": b,
                "Top-k": top_k,
                "Top BM25 Doc": res["top_bm25_doc"],
                "Top Semantic Doc": res["top_semantic_doc"],
                "Top Hybrid Doc": res["top_hybrid_doc"],
                "Overlap@k (%)": round(res["overlap"] * 100, 1),
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="experiment_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_quiz_section():
    st.header("Concept Assessment Quiz")
    st.write("Answer the conceptual questions below to evaluate your understanding of hybrid retrieval.")
    st.caption("No option is pre-selected. Each question must be answered before the quiz can be graded.")

    placeholder = "-- Select an option --"

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            display_options = [placeholder] + q["options"]
            prev_ans = st.session_state["quiz_answers"].get(q["id"])
            default_index = 0 if prev_ans is None else prev_ans + 1
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=display_options,
                index=default_index,
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = None if selected == placeholder else display_options.index(selected) - 1

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        st.session_state["quiz_answers"] = user_responses
        missing = [q["id"] for q in QUIZ_QUESTIONS if user_responses.get(q["id"]) is None]

        if missing:
            st.warning(f"Please answer every question before submitting. Not yet answered: {', '.join(str(i) for i in missing)}")
        else:
            score = 0
            st.session_state["quiz_submitted"] = True

            st.divider()
            st.subheader("Evaluation Results and Feedback")
            for q in QUIZ_QUESTIONS:
                user_ans = user_responses[q["id"]]
                correct_ans = q["answer_index"]
                if user_ans == correct_ans:
                    score += 1
                    st.success(f"Question {q['id']}: Correct!\n\n{q['explanation']}")
                else:
                    st.error(f"Question {q['id']}: Incorrect. (Your answer: {q['options'][user_ans]})\n\n"
                             f"Correct Answer: {q['options'][correct_ans]}\n\n"
                             f"Reasoning: {q['explanation']}")

            st.session_state["quiz_score"] = score
            perc = (score / len(QUIZ_QUESTIONS)) * 100
            st.info(f"Final Score: {score} / {len(QUIZ_QUESTIONS)} ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")


def render_report_section():
    st.header("Report Generation")
    st.write("Compile your student details, recorded trials, and quiz evaluation into an official PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-013"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "Hybrid retrieval was compared against BM25-only and semantic-only rankings across several "
            "queries, showing improved coverage of semantically related but lexically different documents."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"Experiment: {EXPERIMENT_CONFIG['title']}")
    st.write(f"Student: {student_name} | ID: {student_id} | Date: {lab_date}")
    st.write(f"Quiz Score: {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# 5. MAIN ENTRYPOINT AND NAVIGATION

def init_session_state():
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-013",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="Virtual Lab - Hybrid Retrieval",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    st.title(EXPERIMENT_CONFIG["title"])

    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- Quiz Status: {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- Quiz Score: {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()