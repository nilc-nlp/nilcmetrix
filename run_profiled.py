# -*- coding: utf-8 -*-
"""Script para identificar gargalos de performance no NILC-Metrix."""
import text_metrics
import sys
import time
from datetime import datetime

text = sys.argv[1]
raw = text.replace('{{quotes}}', '"')
raw = raw.replace('{{exclamation}}', '!')
raw = raw.replace('{{enter}}', '\n')
raw = raw.replace('{{sharp}}', '#')
raw = raw.replace('{{ampersand}}', '&')
raw = raw.replace('{{percent}}', '%')
raw = raw.replace('{{dollar}}', '$')
raw = raw.encode("utf-8", "surrogateescape").decode("utf-8")

print("=" * 60)
print("PROFILING NILC-Metrix")
print("=" * 60)

start = time.time()
t = text_metrics.Text(raw)
print(f"[{time.time()-start:.3f}s] Criar objeto Text")

rp = text_metrics.rp

start = time.time()
sentences = rp.sentences(t)
print(f"[{time.time()-start:.3f}s] Tokenizar sentenças ({len(sentences)} sentenças)")

start = time.time()
tokens = rp.tokens(t)
print(f"[{time.time()-start:.3f}s] Tokenizar palavras")

start = time.time()
tagged = rp.tagged_sentences(t)
print(f"[{time.time()-start:.3f}s] POS Tagging")

start = time.time()
parse_trees = rp.parse_trees(t)
print(f"[{time.time()-start:.3f}s] Parse Trees (LX-Parser) - PRIMEIRA CHAMADA")

start = time.time()
parse_trees2 = rp.parse_trees(t)
print(f"[{time.time()-start:.3f}s] Parse Trees - SEGUNDA CHAMADA (cache)")

start = time.time()
dep_trees = rp.dep_trees(t)
print(f"[{time.time()-start:.3f}s] Dependency Trees (MaltParser) - PRIMEIRA CHAMADA")

start = time.time()
dep_trees2 = rp.dep_trees(t)
print(f"[{time.time()-start:.3f}s] Dependency Trees - SEGUNDA CHAMADA (cache)")

print("\n" + "=" * 60)
print("PROFILE POR CATEGORIA DE MÉTRICAS")
print("=" * 60)

categories = [
    ("BasicCounts", text_metrics.metrics.BasicCounts()),
    ("LogicOperators", text_metrics.metrics.LogicOperators()),
    ("Frequencies", text_metrics.metrics.Frequencies()),
    ("Hypernyms", text_metrics.metrics.Hypernyms()),
    ("Tokens", text_metrics.metrics.Tokens()),
    ("Connectives", text_metrics.metrics.Connectives()),
    ("Ambiguity", text_metrics.metrics.Ambiguity()),
    ("Anaphoras", text_metrics.metrics.Anaphoras()),
    ("Coreference", text_metrics.metrics.Coreference()),
    ("Lsa", text_metrics.metrics.Lsa()),
    ("LIWC", text_metrics.metrics.LIWC()),
    ("EXTRA", text_metrics.metrics.extra.EXTRA()),
]

syntax_cat = text_metrics.base.Category([
    text_metrics.metrics.syntax.YngveComplexity(),
    text_metrics.metrics.syntax.FrazierComplexity(),
    text_metrics.metrics.syntax.DependencyDistance(),
    text_metrics.metrics.syntax.CrossEntropy()
], name='Syntactical Complexity', table_name='syntax')
categories.append(("SyntacticalComplexity", syntax_cat))

content_cat = text_metrics.base.Category([
    text_metrics.metrics.sem.ContentDensity()
], name='Semantic Density', table_name='semantic_density')
categories.append(("ContentDensity", content_cat))

total_time = 0
for name, category in categories:
    start = time.time()
    try:
        result = category.values_for_text(t)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"[{elapsed:.3f}s] {name}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{elapsed:.3f}s] {name} - ERRO: {e}")

print("\n" + "=" * 60)
print(f"TEMPO TOTAL DAS CATEGORIAS: {total_time:.3f}s")
print("=" * 60)
