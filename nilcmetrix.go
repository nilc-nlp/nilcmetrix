package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os/exec"
	"sort"
	"strings"
	"text/template"
	"time"

	"github.com/gorilla/mux"
)

type Metric struct {
	Source string `json:"s"`
	Level  string `json:"l"`
	Class  string `json:"c"`
	Key    string `json:"k"`
	Desc   string `json:"d"`
}

var metricMap = map[string]Metric{
	"adjacent_refs":                       {"CMP2", "Texto", "Coesão Referencial", "adjacent_refs", "Média das proporções  de candidatos a referentes na sentença anterior em relação aos pronomes pessoais do caso reto nas sentenças"},
	"anaphoric_refs":                      {"CMP2", "Texto", "Coesão Referencial", "anaphoric_refs", "Média das proporções  de candidatos a referentes nas 5 sentenças anteriores em relação aos pronomes anafóricos das sentenças"},
	"adj_arg_ovl":                         {"CMP2", "Texto", "Coesão Referencial", "adj_arg_ovl", "Quantidade média de referentes que se repetem nos pares de sentenças adjacentes do texto"},
	"adj_cw_ovl":                          {"CMP2", "Texto", "Coesão Referencial", "adj_cw_ovl", "Quantidade média de palavras de conteúdo que se repetem nos pares de sentenças adjacentes do texto"},
	"adj_stem_ovl":                        {"CMP2", "Texto", "Coesão Referencial", "adj_stem_ovl", "Quantidade média de radicais de palavras de conteúdo que se repetem nos pares de sentenças adjacentes do texto"},
	"arg_ovl":                             {"CMP2", "Texto", "Coesão Referencial", "arg_ovl", "Quantidade média de referentes que se repetem nos pares de sentenças do texto"},
	"stem_ovl":                            {"CMP2", "Texto", "Coesão Referencial", "stem_ovl", "Quantidade média de radicais de palavras de conteúdo que se repetem nos pares de sentenças do texto"},
	"coreference_pronoun_ratio":           {"GTN", "Sentença", "coesão Referencial", "coreference_pronoun_ratio", "Média de candidatos a referente (na sentença anterior) por pronome anafórico do caso reto"},
	"demonstrative_pronoun_ratio":         {"GTN", "Sentença", "coesão Referencial", "demonstrative_pronoun_ratio", "Média de candidatos a referente (na sentença anterior) por pronome demonstrativo anafórico"},
	"lsa_adj_mean":                        {"CMD", "Texto", "Coesão Semântica", "lsa_adj_mean", "Média de similaridade entre pares de sentenças adjacentes no texto"},
	"lsa_adj_std":                         {"CMD", "Texto", "Coesão Semântica", "lsa_adj_std", "Desvio padrão de similaridade entre pares de sentenças adjacentes no texto"},
	"lsa_all_mean":                        {"CMD", "Texto", "Coesão Semântica", "lsa_all_mean", "Média de similaridade entre todos os pares de sentenças no texto"},
	"lsa_all_std":                         {"CMD", "Texto", "Coesão Semântica", "lsa_all_std", "Desvio padrão de similaridade entre todos os pares possíveis de sentenças do texto"},
	"lsa_givenness_mean":                  {"CMD", "Texto", "Coesão Semântica", "lsa_givenness_mean", "Média do *givenness* da cada sentença do texto a partir da segunda"},
	"lsa_givenness_std":                   {"CMD", "Texto", "Coesão Semântica", "lsa_givenness_std", "Desvio padrão do *givenness* da cada sentença do texto a partir da segunda"},
	"lsa_paragraph_mean":                  {"CMD", "Texto", "Coesão Semântica", "lsa_paragraph_mean", "Média de similaridade entre pares de parágrafos adjacentes no texto"},
	"lsa_paragraph_std":                   {"CMD", "Texto", "Coesão Semântica", "lsa_paragraph_std", "Desvio padrão entre parágrafos adjacentes no texto"},
	"lsa_span_mean":                       {"CMD", "Texto", "Coesão Semântica", "lsa_span_mean", "Média do *span* da cada sentença do texto a partir da segunda"},
	"lsa_span_std":                        {"CMD", "Texto", "Coesão Semântica", "lsa_span_std", "Desvio padrão do span da cada sentença do texto a partir da segunda"},
	"cross_entropy":                       {"CMD", "Texto", "Coesão Semântica", "cross_entropy", "Média da entropia cruzadas das sentenças do texto"},
	"words_before_main_verb":              {"CMP", "Sentença", "Complexidade Sintática", "words_before_main_verb", "Quantidade Média de palavras antes dos verbos principais das orações principais das sentenças"},
	"adjunct_per_clause":                  {"AIC", "Sentença", "Complexidade Sintática", "adjunct_per_clause", "Quantidade média de adjuntos adverbiais por oração do texto"},
	"adverbs_before_main_verb_ratio":      {"GTN", "Sentença", "Complexidade Sintática", "adverbs_before_main_verb_ratio", "Proporção de orações com advérbio antes do verbo principal em relação à quantidade de orações do texto"},
	"apposition_per_clause":               {"AIC", "Sentença", "Complexidade Sintática", "apposition_per_clause", "Quantidade média de apostos por oração do texto"},
	"clauses_per_sentence":                {"AIC", "Sentença", "Complexidade Sintática", "clauses_per_sentence", "Quantidade média de orações por sentença"},
	"coordinate_conjunctions_per_clauses": {"AIC", "Sentença", "Complexidade Sintática", "coordinate_conjunctions_per_clauses", "Proporção de conjunções coordenativas em relação a todas as orações do texto"},
	"dep_distance":                        {"CMD", "Sentença", "Complexidade Sintática", "dep_distance", "Distância na árvore de dependências"},
	"frazier":                             {"CMD", "Sentença", "Complexidade Sintática", "frazier", "Fórmula de Complexidade Sintática de Frazier"},
	"infinite_subordinate_clauses":        {"GTN", "Sentença", "Complexidade Sintática", "infinite_subordinate_clauses", "Proporção de orações subordinadas reduzidas pela quantidade de orações do texto"},
	"non_svo_ratio":                       {"GTN", "Sentença", "Complexidade Sintática", "non_svo_ratio", "Proporção de orações que não estão no formato SVO (sujeito-verbo-objeto) em relação a todas orações do texto"},
	"passive_ratio":                       {"AIC", "Sentença", "Complexidade Sintática", "passive_ratio", "Proporção de orações na voz passiva analítica em relação à quantidade de orações do texto"},
	"postponed_subject_ratio":             {"GTN", "Sentença", "Complexidade Sintática", "postponed_subject_ratio", "Proporção de sujeitos pospostos em relação a todos os sujeitos do texto"},
	"ratio_coordinate_conjunctions":       {"AIC", "Sentença", "Complexidade Sintática", "ratio_coordinate_conjunctions", "Proporção de conjunções coordenativas em relação a todas as conjunções do texto"},
	"ratio_subordinate_conjunctions":      {"AIC", "Sentença", "Complexidade Sintática", "ratio_subordinate_conjunctions", "Proporção de conjunções subordinativas em relação a todas conjunções do texto"},
	"relative_clauses":                    {"AIC", "Sentença", "Complexidade Sintática", "relative_clauses", "Proporção de orações relativas em relação à quantidade de orações do texto"},
	"sentences_with_five_clauses":         {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_five_clauses", "Proporção de sentenças com 5 orações em relação a todas as sentenças do texto"},
	"sentences_with_four_clauses":         {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_four_clauses", "Proporção de sentenças com 4 orações em relação a todas as sentenças do texto"},
	"sentences_with_one_clause":           {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_one_clause", "Proporção de sentenças com 1 oração em relação a todas as sentenças do texto"},
	"sentences_with_seven_more_clauses":   {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_seven_more_clauses", "Proporção de sentenças com 7 ou mais orações em relação a todas as sentenças do texto"},
	"sentences_with_six_clauses":          {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_six_clauses", "Proporção de sentenças com 6 orações em relação a todas as sentenças do texto"},
	"sentences_with_three_clauses":        {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_three_clauses", "Proporção de sentenças com 3 orações em relação a todas as sentenças do texto"},
	"sentences_with_two_clauses":          {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_two_clauses", "Proporção de sentenças com 2 orações em relação a todas as sentenças do texto"},
	"sentences_with_zero_clause":          {"AIC", "Sentença", "Complexidade Sintática", "sentences_with_zero_clause", "Proporção de sentenças sem verbos em relação a todas as sentenças do texto"},
	"std_noun_phrase":                     {"GTN", "Sentença", "Complexidade Sintática", "std_noun_phrase", "Desvio-padrão do tamanho dos sintagmas nominais do texto"},
	"subordinate_clauses":                 {"GTN", "Sentença", "Complexidade Sintática", "subordinate_clauses", "Proporção de orações subordinadas pela quantidade de orações do texto"},
	"temporal_adjunct_ratio":              {"GTN", "Sentença", "Complexidade Sintática", "temporal_adjunct_ratio", "Proporção de adjuntos adverbiais de tempo em relação a todos os adjuntos adverbiais do texto"},
	"yngve":                               {"CMD", "Sentença", "Complexidade Sintática", "yngve", "Fórmula de Complexidade Sintática de Yngve"},
	"add_neg_conn_ratio":                  {"CMP", "Texto", "Conectivos", "add_neg_conn_ratio", "Proporção de conectivos  aditivos negativos em relação à quantidade de palavras do texto"},
	"add_pos_conn_ratio":                  {"CMP", "Texto", "Conectivos", "add_pos_conn_ratio", "Proporção de conectivos  aditivos positivos em relação à quantidade de palavras do texto"},
	"cau_neg_conn_ratio":                  {"CMP", "Texto", "Conectivos", "cau_neg_conn_ratio", "Proporção de conectivos  causais negativos em relação à quantidade de palavras do texto"},
	"cau_pos_conn_ratio":                  {"CMP", "Texto", "Conectivos", "cau_pos_conn_ratio", "Proporção de conectivos  causais positivos em relação à quantidade de palavras do texto"},
	"conn_ratio":                          {"CMP", "Texto", "Conectivos", "conn_ratio", "Proporção de Conectivos em relação à quantidade de palavras do texto"},
	"log_neg_conn_ratio":                  {"CMP", "Texto", "Conectivos", "log_neg_conn_ratio", "Proporção de Conectivos Lógicos Negativos em relação à quantidade de palavras do texto"},
	"log_pos_conn_ratio":                  {"CMP", "Texto", "Conectivos", "log_pos_conn_ratio", "Proporção de Conectivos Lógicos Positivos em relação à quantidade de palavras do texto"},
	"tmp_neg_conn_ratio":                  {"CMP", "Texto", "Léxico Temporal", "tmp_neg_conn_ratio", "Proporção de conectivos  temporais negativos em relação à quantidade de palavras do texto"},
	"tmp_pos_conn_ratio":                  {"CMP", "Texto", "Léxico Temporal", "tmp_pos_conn_ratio", "Proporção de conectivos  temporais positivosem relação à quantidade de palavras do texto"},
	"and_ratio":                           {"CMP", "Texto", "Conectivos", "and_ratio", "Proporção do operador lógico E em relação à quantidade de palavras do texto"},
	"if_ratio":                            {"CMP", "Texto", "Conectivos", "if_ratio", "Proporção do operador lógico SE em relação à quantidade de palavras do texto"},
	"logic_operators":                     {"CMP", "Texto", "Conectivos", "logic_operators", "Proporção de Operadores Lógicos em relação à quantidade de palavras do texto"},
	"negation_ratio":                      {"CMP", "Texto", "Conectivos", "negation_ratio", "Proporção de palavras que denotam negação em relação à quantidade de palavras do texto"},
	"or_ratio":                            {"CMP", "Texto", "Conectivos", "or_ratio", "Proporção do operador lógico OU em relação à quantidade de palavras do texto"},
	"mean_noun_phrase":                    {"CMP2", "Sentença", "Densidade de Padrões Sintáticos", "mean_noun_phrase", "Média dos tamanhos médios dos sintagmas nominais nas sentenças"},
	"aux_plus_PCP_per_sentence":           {"AIC", "Sentença", "Léxico Temporal", "aux_plus_PCP_per_sentence", "Proporção de verbos auxiliares seguidos de particípio em relação à quantidade de sentenças do texto"},
	"gerund_verbs":                        {"AIC", "Sentença", "Densidade de Padrões Sintáticos", "gerund_verbs", "Proporção de verbos no gerúndio em relação a todos os verbos do texto"},
	"max_noun_phrase":                     {"GTN", "Sentença", "Densidade de Padrões Sintáticos", "max_noun_phrase", "Máximo entre os tamanhos de sintagmas nominais do texto"},
	"min_noun_phrase":                     {"GTN", "Sentença", "Densidade de Padrões Sintáticos", "min_noun_phrase", "Mínimo entre os tamanhos de sintagmas nominais do texto"},
	"ttr":                                 {"CMP", "Texto", "Diversidade Lexical", "ttr", "Proporção de types (despreza repetições de palavras) em relação à quantidade de tokens (computa repetições de palavras) no texto"},
	"adjective_diversity_ratio":           {"GTN", "Sentença", "Diversidade Lexical", "adjective_diversity_ratio", "Proporção de types de adjetivos  em relação à quantidade de tokens de adjetivos no texto"},
	"content_density":                     {"GTN", "Sentença", "Diversidade Lexical", "content_density", "Proporção de palavras de conteúdo em relação à quantidade de palavras funcionais do texto"},
	"content_word_diversity":              {"GTN", "Sentença", "Diversidade Lexical", "content_word_diversity", "Proporção de types de palavras de conteúdo em relação à quantidade de tokens de palavras de conteúdo no texto"},
	"content_word_max":                    {"GTN", "Sentença", "Diversidade Lexical", "content_word_max", "Proporção máxima de palavras de conteúdo em relação à quantidade de palavras das sentenças"},
	"content_word_min":                    {"GTN", "Sentença", "Diversidade Lexical", "content_word_min", "Proporção Mínima de palavras de conteúdo por quantidade de palavras nas sentenças"},
	"content_word_standard_deviation":     {"GTN", "Sentença", "Diversidade Lexical", "content_word_standard_deviation", "Desvio padrão das proporções entre as palavras de conteúdo e a quantidade de palavras das sentenças"},
	"function_word_diversity":             {"GTN", "Sentença", "Diversidade Lexical", "function_word_diversity", "Proporção  de types de palavras funcionais em relação à quantidade de tokens de palavras funcionais no texto"},
	"indefinite_pronouns_diversity":       {"GTN", "Sentença", "Diversidade Lexical", "indefinite_pronouns_diversity", "Proporção  de types de pronomes indefinidos em relação à quantidade de tokens de  pronomes indefinidos no texto"},
	"noun_diversity":                      {"GTN", "Sentença", "Diversidade Lexical", "noun_diversity", "Proporção de types de substantivos em relação à quantidade de tokens de  substantivos no texto"},
	"preposition_diversity":               {"GTN", "Sentença", "Diversidade Lexical", "preposition_diversity", "Proporção  de types de preposições em relação à quantidade de tokens de preposições no texto"},
	"pronoun_diversity":                   {"GTN", "Sentença", "Diversidade Lexical", "pronoun_diversity", "Proporção de types de pronomes em relação à quantidade de tokens de  pronomes no texto"},
	"punctuation_diversity":               {"GTN", "Texto", "Diversidade Lexical", "punctuation_diversity", "Proporção  de types de pontuações em relação à quantidade de tokens de pontuações no texto"},
	"relative_pronouns_diversity_ratio":   {"GTN", "Sentença", "Diversidade Lexical", "relative_pronouns_diversity_ratio", "Proporção  de types de  pronomes relativos  em relação à quantidade de tokens de  pronomes relativos no texto"},
	"verb_diversity":                      {"GTN", "Sentença", "Diversidade Lexical", "verb_diversity", "Proporção de types de verbos em relação à quantidade de tokens de verbos no texto"},
	"cw_freq":                             {"CMP", "Sentença", "Frequência de Palavras", "cw_freq", "Média das frequências absolutas das palavras de conteúdo do texto"},
	"cw_freq_bra":                         {"RTS", "Sentença", "Frequência de Palavras", "cw_freq_bra", "Média dos valores das frequências das palavras de conteúdo do texto na escala logarítmica Zipf via Corpus Brasileiro"},
	"cw_freq_brwac":                       {"RTS", "Sentença", "Frequência de Palavras", "cw_freq_brwac", "Média dos valores das frequências das palavras de conteúdo do texto na escala logarítmica Zipf via BrWac"},
	"freq_bra":                            {"RTS", "Palavra", "Frequência de Palavras", "freq_bra", "Média dos valores das frequências das palavras do texto na escala logarítmica Zipf via Corpus Brasileiro"},
	"freq_brwac":                          {"RTS", "Palavra", "Frequência de Palavras", "freq_brwac", "Média dos valores das frequências das palavras do texto na escala logarítmica Zipf via BrWac"},
	"min_cw_freq":                         {"CMP", "Sentença", "Frequência de Palavras", "min_cw_freq", "Média das frequências das palavras de conteúdo mais raras das sentenças do texto"},
	"min_cw_freq_bra":                     {"RTS", "Sentença", "Frequência de Palavras", "min_cw_freq_bra", "Média dos valores das frequências das palavras de conteúdo mais raras das sentenças do texto na escala logarítmica Zipf via Corpus Brasileiro"},
	"min_cw_freq_brwac":                   {"RTS", "Sentença", "Frequência de Palavras", "min_cw_freq_brwac", "Média dos valores das frequências das palavras de conteúdo do texto na escala logarítmica Zipf via BrWac"},
	"min_freq_bra":                        {"RTS", "Palavra", "Frequência de Palavras", "min_freq_bra", "Média dos valores das frequências das palavras mais raras das sentenças do texto na escala logarítmica Zipf via Corpus Brasileiro"},
	"min_freq_brwac":                      {"RTS", "Palavra", "Frequência de Palavras", "min_freq_brwac", "Média dos valores das frequências das palavras mais raras das sentenças do texto na escala logarítmica Zipf via BrWac"},
	"brunet":                              {"CMD", "Texto", "Índices de Leiturabilidade", "brunet", "Índice de Brunet"},
	"dalechall_adapted":                   {"GTN", "Texto", "Índices de Leiturabilidade", "dalechall_adapted", "Fórmula Dale Chall adaptada"},
	"flesch":                              {"CMP", "Texto", "Índices de Leiturabilidade", "flesch", "Índice Flesch"},
	"gunning_fox":                         {"GTN", "Texto", "Índices de Leiturabilidade", "gunning_fox", "Índice Gunning Fog"},
	"honore":                              {"CMD", "Texto", "Índices de Leiturabilidade", "honore", "Estatística de Honoré"},
	"adjective_ratio":                     {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "adjective_ratio", "Proporção de Adjetivos em relação à quantidade de palavras do texto"},
	"adverbs":                             {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "adverbs", "Proporção de Advérbios em relação à quantidade de palavras do texto"},
	"content_words":                       {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "content_words", "Proporção de palavras de conteúdo em relação à quantidade de palavras do texto"},
	"function_words":                      {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "function_words", "Proporção de Palavras Funcionais em relação à quantidade de palavras do texto"},
	"noun_ratio":                          {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "noun_ratio", "Proporção de substantivos em relação à quantidade de palavras do texto"},
	"pronoun_ratio":                       {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "pronoun_ratio", "Proporção de pronomes em relação à quantidade de palavras do texto"},
	"verbs":                               {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "verbs", "Proporção de Verbos em relação à quantidade de palavras do texto"},
	"personal_pronouns":                   {"CMP", "Sentença", "Informações Morfossintáticas de Palavras", "personal_pronouns", "Proporção de Pronomes Pessoais em relação à quantidade de palavras do texto"},
	"adjectives_max":                      {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adjectives_max", "Proporção máxima de adjetivos em relação à quantidade de palavras das sentenças"},
	"adjectives_min":                      {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adjectives_min", "Proporção mínima de adjetivos em relação à quantidade de palavras das sentenças"},
	"adjectives_standard_deviation":       {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adjectives_standard_deviation", "Desvio padrão das proporções entre adjetivos e a quantidade de palavras das sentenças"},
	"adverbs_diversity_ratio":             {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adverbs_diversity_ratio", "Proporção  de types de advérbios em relação à quantidade de tokens de  advérbios no texto"},
	"adverbs_max":                         {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adverbs_max", "Proporção máxima de advérbios em relação à quantidade de palavras das sentenças"},
	"adverbs_min":                         {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adverbs_min", "Proporção mínima de advérbios em relação à quantidade de palavras das sentenças"},
	"adverbs_standard_deviation":          {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "adverbs_standard_deviation", "Desvio padrão das proporções entre advérbios e a quantidade de palavras das sentenças"},
	"indefinite_pronoun_ratio":            {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "indefinite_pronoun_ratio", "Proporção de pronomes indefinidos em relação a todos os pronomes do texto"},
	"indicative_condition_ratio":          {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "indicative_condition_ratio", "Proporção de Verbos no Futuro do Pretérito do Indicativo em relação à quantidade de verbos flexionados do texto"},
	"indicative_future_ratio":             {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "indicative_future_ratio", "Proporção de Verbos no Futuro do Presente do Indicativo em relação à quantidade de verbos flexionados do texto"},
	"indicative_imperfect_ratio":          {"GTN", "Sentença", "Léxico Temporal", "indicative_imperfect_ratio", "Proporção de Verbos no Pretérito Imperfeito do Indicativo em relação à quantidade de verbos flexionados no texto"},
	"indicative_pluperfect_ratio":         {"GTN", "Sentença", "Léxico Temporal", "indicative_pluperfect_ratio", "Proporção de Verbos no Pretérito Mais que Perfeito do Indicativo em relação à quantidade de verbos flexionados no texto"},
	"indicative_present_ratio":            {"GTN", "Sentença", "Léxico Temporal", "indicative_present_ratio", "Proporção de Verbos no Presente do Indicativo em relação à quantidade de verbos flexionados no texto"},
	"indicative_preterite_perfect_ratio":  {"GTN", "Sentença", "Léxico Temporal", "indicative_preterite_perfect_ratio", "Proporção de Verbos no Pretérito Perfeito Simples do Indicativo em relação à quantidade de verbos flexionados no texto"},
	"infinitive_verbs":                    {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "infinitive_verbs", "Proporção de verbos no infinitivo em relação a todos os verbos do texto"},
	"inflected_verbs":                     {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "inflected_verbs", "Proporção de verbos flexionados em relação a todos os verbos do texto"},
	"non-inflected_verbs":                 {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "non-inflected_verbs", "Proporção de verbos no gerúndio/particípio/infinitivo em relação a todos os verbos do texto"},
	"nouns_max":                           {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "nouns_max", "Proporção máxima de substantivos em relação à quantidade de palavras das sentenças"},
	"nouns_min":                           {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "nouns_min", "Proporção mínima de substantivos em relação à quantidade de palavras das sentenças"},
	"nouns_standard_deviation":            {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "nouns_standard_deviation", "Desvio padrão das proporções entre substantivos e a quantidade de palavras das sentenças"},
	"oblique_pronouns_ratio":              {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "oblique_pronouns_ratio", "Proporção de pronomes oblíquos em relação a todos os pronomes do texto"},
	"participle_verbs":                    {"AIC", "Sentença", "Léxico Temporal", "participle_verbs", "Proporção de verbos no particípio em relação a todos os verbos do texto"},
	"prepositions_per_clause":             {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "prepositions_per_clause", "Proporção de preposições em relação à quantidade de orações no texto"},
	"prepositions_per_sentence":           {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "prepositions_per_sentence", "Quantidade Média de preposições por sentença no texto"},
	"pronouns_max":                        {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "pronouns_max", "Proporção máxima de pronomes em relação à quantidade de palavras das sentenças"},
	"pronouns_min":                        {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "pronouns_min", "Proporção mínima de pronomes em relação à quantidade de palavras das sentenças"},
	"pronouns_standard_deviation":         {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "pronouns_standard_deviation", "Desvio padrão das proporções entre pronomes e a quantidade de palavras das sentenças"},
	"punctuation_ratio":                   {"GTN", "Texto", "Informações Morfossintáticas de Palavras", "punctuation_ratio", "Proporção de sinais de pontuação em relação à quantidade de palavras do texto"},
	"ratio_function_to_content_words":     {"AIC", "Sentença", "Informações Morfossintáticas de Palavras", "ratio_function_to_content_words", "Proporção de palavras funcionais em relação à quantidade de palavras de conteúdo do texto"},
	"relative_pronouns_ratio":             {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "relative_pronouns_ratio", "Proporção de Pronomes Relativos em relação à quantidade de pronomes do texto"},
	"second_person_possessive_pronouns":   {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "second_person_possessive_pronouns", "Proporção de pronomes possessivos nas segundas pessoas em relação à quantidade de pronomes possessivos do texto"},
	"second_person_pronouns":              {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "second_person_pronouns", "Proporção de pronomes pessoais nas segundas pessoas em relação à quantidade de pronomes pessoais do texto"},
	"third_person_possessive_pronouns":    {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "third_person_possessive_pronouns", "Proporção de pronomes possessivos nas terceiras pessoas em relação à quantidade de pronomes possessivos do texto"},
	"third_person_pronouns":               {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "third_person_pronouns", "Proporção de pronomes pessoais nas terceiras pessoas em relação à quantidade de pronomes pessoais do texto"},
	"verbs_max":                           {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "verbs_max", "Proporção máxima de verbos em relação à quantidade de palavras das sentenças"},
	"verbs_min":                           {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "verbs_min", "Proporção mínima de verbos em relação à quantidade de palavras das sentenças"},
	"verbs_standard_deviation":            {"GTN", "Sentença", "Informações Morfossintáticas de Palavras", "verbs_standard_deviation", "Desvio padrão das proporções entre verbos e a quantidade de palavras das sentenças"},
	"adjectives_ambiguity":                {"CMP", "Sentença", "Informações Semânticas de Palavras", "adjectives_ambiguity", "Proporção de sentidos dos adjetivos do texto em relação à quantidade de adjetivos do texto"},
	"adverbs_ambiguity":                   {"CMP", "Sentença", "Informações Semânticas de Palavras", "adverbs_ambiguity", "Proporção de sentidos dos advérbios do texto em relação à  quantidade de advérbios do texto"},
	"nouns_ambiguity":                     {"CMP", "Sentença", "Informações Semânticas de Palavras", "nouns_ambiguity", "Proporção de sentidos dos substantivos do texto em relação à  quantidade de substantivos do texto"},
	"verbs_ambiguity":                     {"CMP", "Sentença", "Informações Semânticas de Palavras", "verbs_ambiguity", "Proporção de sentidos dos verbos do texto em relação à quantidade de verbos do texto"},
	"hypernyms_verbs":                     {"CMP", "Sentença", "Informações Semânticas de Palavras", "hypernyms_verbs", "Quantidade Média de Hiperônimos por verbo nas sentenças"},
	"abstract_nouns_ratio":                {"GTN", "Sentença", "Informações Semânticas de Palavras", "abstract_nouns_ratio", "Proporção de substantivos abstratos em relação à quantidade de palavras do texto"},
	"content_words_ambiguity":             {"GTN", "Sentença", "Informações Semânticas de Palavras", "content_words_ambiguity", "Média de sentidos por palavra de conteúdo do texto"},
	"named_entity_ratio_sentence":         {"GTN", "Sentença", "Informações Semânticas de Palavras", "named_entity_ratio_sentence", "Média das proporções de Nomes Próprios em relação à quantidade de palavras das Sentenças"},
	"named_entity_ratio_text":             {"GTN", "Sentença", "Informações Semânticas de Palavras", "named_entity_ratio_text", "Proporção de Nomes Próprios em relação à quantidade de palavras do Texto"},
	"negative_words":                      {"CMD", "Palavra", "Informações Semânticas de Palavras", "negative_words", "Proporção de palavras de polaridade negativa em relação a todas palavras do texto"},
	"positive_words":                      {"CMD", "Palavra", "Informações Semânticas de Palavras", "positive_words", "Proporção de palavras de polaridade positiva em relação a todas palavras do texto"},
	"paragraphs":                          {"CMP", "Texto", "Medidas Descritivas", "paragraphs", "Quantidade de Parágrafos no texto"},
	"sentences":                           {"CMP", "Texto", "Medidas Descritivas", "sentences", "Quantidade de Sentenças no texto"},
	"sentences_per_paragraph":             {"CMP", "Texto", "Medidas Descritivas", "sentences_per_paragraph", "Quantidade média de sentenças por parágrafo no texto"},
	"syllables_per_content_word":          {"CMP", "Sentença", "Medidas Descritivas", "syllables_per_content_word", "Quantidade média de sílabas por palavra de conteúdo no texto"},
	"words":                               {"CMP", "Texto", "Medidas Descritivas", "words", "Quantidade de Palavras no texto"},
	"words_per_sentence":                  {"CMP", "Sentença", "Medidas Descritivas", "words_per_sentence", "Média de Palavras por Sentença"},
	"sentence_length_max":                 {"GTN", "Sentença", "Medidas Descritivas", "sentence_length_max", "Quantidade Máxima de palavras por sentença"},
	"sentence_length_min":                 {"GTN", "Sentença", "Medidas Descritivas", "sentence_length_min", "Quantidade Mínima de palavras por sentença"},
	"sentence_length_standard_deviation":  {"GTN", "Sentença", "Medidas Descritivas", "sentence_length_standard_deviation", "Desvio Padrão da quantidade de palavras por sentença"},
	"subtitles":                           {"GTN", "Texto", "Medidas Descritivas", "subtitles", "Proporção de Subtítulos em relação à quantidade de sentenças do texto"},
	"concretude_1_25_ratio":               {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_1_25_ratio", "Proporção de palavras com valor de concretude entre 1 e 2.5  em relação a todas as palavras de conteúdo do texto"},
	"concretude_25_4_ratio":               {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_25_4_ratio", "Proporção de palavras com valor de concretude entre 2.5  e 4 em relação a todas as palavras de conteúdo do texto"},
	"concretude_4_55_ratio":               {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_4_55_ratio", "Proporção de palavras com valor de concretude entre 4 e 5.5  em relação a todas as palavras de conteúdo do texto"},
	"concretude_55_7_ratio":               {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_55_7_ratio", "Proporção de palavras com valor de concretude entre 5.5 e 7  em relação a todas as palavras de conteúdo do texto"},
	"concretude_mean":                     {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_mean", "Média dos valores de concretude das palavras de conteúdo do texto"},
	"concretude_std":                      {"GTN", "Palavra", "Medidas Psicolinguísticas", "concretude_std", "Desvio padrão do valor de concretude das palavras de conteúdo do texto"},
	"familiaridade_1_25_ratio":            {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_1_25_ratio", "Proporção de palavras com valor de familiaridade entre 1 e 2.5  em relação a todas as palavras de conteúdo do texto"},
	"familiaridade_25_4_ratio":            {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_25_4_ratio", "Proporção de palavras com valor de familiaridade entre 2.5  e 4 em relação a todas as palavras de conteúdo do texto"},
	"familiaridade_4_55_ratio":            {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_4_55_ratio", "Proporção de palavras com valor de familiaridade entre 4 e 5.5  em relação a todas as palavras de conteúdo do texto"},
	"familiaridade_55_7_ratio":            {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_55_7_ratio", "Proporção de palavras com valor de familiaridade entre 5.5 e 7  em relação a todas as palavras de conteúdo do texto"},
	"familiaridade_mean":                  {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_mean", "Média dos valores de familiaridade das palavras de conteúdo do texto"},
	"familiaridade_std":                   {"GTN", "Palavra", "Medidas Psicolinguísticas", "familiaridade_std", "Desvio padrão dos valores de familiaridade das palavras de conteúdo do texto"},
	"idade_aquisicao_1_25_ratio":          {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_1_25_ratio", "Proporção de palavras com valor de idade de aquisição entre 1 e 2.5 em relação a todas as palavras de conteúdo do texto"},
	"idade_aquisicao_25_4_ratio":          {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_25_4_ratio", "Proporção de palavras com valor de idade de aquisição entre 2.5  e 4 em relação a todas as palavras de conteúdo do texto"},
	"idade_aquisicao_4_55_ratio":          {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_4_55_ratio", "Proporção de palavras com valor de idade de aquisição entre 4 e 5.5 em relação a todas as palavras de conteúdo do texto"},
	"idade_aquisicao_55_7_ratio":          {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_55_7_ratio", "Proporção de palavras com valor de idade de aquisição entre 5.5 e 7 em relação a todas as palavras de conteúdo do texto"},
	"idade_aquisicao_mean":                {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_mean", "Média dos valores de idade de aquisição das palavras de conteúdo do texto"},
	"idade_aquisicao_std":                 {"GTN", "Palavra", "Medidas Psicolinguísticas", "idade_aquisicao_std", "Desvio padrão dos valores de idade de aquisição das palavras de conteúdo do texto"},
	"imageabilidade_1_25_ratio":           {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_1_25_ratio", "Proporção de palavras com valor de imageabilidade entre 1 e 2.5  em relação a todas as palavras de conteúdo do texto"},
	"imageabilidade_25_4_ratio":           {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_25_4_ratio", "Proporção de palavras com valor de imageabilidade entre 2.5  e 4 em relação a todas as palavras de conteúdo do texto"},
	"imageabilidade_4_55_ratio":           {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_4_55_ratio", "Proporção de palavras com valor de imageabilidade entre 4 e 5.5  em relação a todas as palavras de conteúdo do texto"},
	"imageabilidade_55_7_ratio":           {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_55_7_ratio", "Proporção de palavras com valor de imageabilidade entre 5.5 e 7  em relação a todas as palavras de conteúdo do texto"},
	"imageabilidade_mean":                 {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_mean", "Média dos valores de imageabilidade das palavras de conteúdo do texto"},
	"imageabilidade_std":                  {"GTN", "Palavra", "Medidas Psicolinguísticas", "imageabilidade_std", "Desvio padrão dos valores de imageabilidade das palavras de conteúdo do texto"},
	"first_person_possessive_pronouns":    {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "first_person_possessive_pronouns", "Proporção de pronomes possessivos nas primeiras pessoas em relação à quantidade de pronomes possessivos do texto"},
	"first_person_pronouns":               {"AIC", "Palavra", "Informações Morfossintáticas de Palavras", "first_person_pronouns", "Proporção de pronomes pessoais nas primeiras pessoas em relação à quantidade de pronomes pessoais do texto"},
	"verbal_time_moods_diversity":         {"GTN", "Sentença", "Léxico Temporal", "verbal_time_moods_diversity", "Quantidade de diferentes tempos-modos verbais que ocorrem no texto"},
	"dialog_pronoun_ratio":                {"GTN", "Palavra", "Simplicidade Textual", "dialog_pronoun_ratio", "Proporção de pronomes pessoais que indicam uma conversa com o leitor em relação à quantidade de pronomes pessoais do texto"},
	"easy_conjunctions_ratio":             {"GTN", "Palavra", "Simplicidade Textual", "easy_conjunctions_ratio", "Proporção de conjunções fáceis em relação à quantidade de palavras do texto"},
	"hard_conjunctions_ratio":             {"GTN", "Palavra", "Simplicidade Textual", "hard_conjunctions_ratio", "Proporção de conjunções difíceis em relação à quantidade de palavras do texto"},
	"long_sentence_ratio":                 {"GTN", "Sentença", "Simplicidade Textual", "long_sentence_ratio", "Proporção de Sentenças Muito Longas em relação a todas as sentenças do texto"},
	"medium_long_sentence_ratio":          {"GTN", "Sentença", "Simplicidade Textual", "medium_long_sentence_ratio", "Proporção de Sentenças Longas em relação a todas as sentenças do texto"},
	"medium_short_sentence_ratio":         {"GTN", "Sentença", "Simplicidade Textual", "medium_short_sentence_ratio", "Proporção de Sentenças Médias em relação a todas as sentenças do texto"},
	"short_sentence_ratio":                {"GTN", "Sentença", "Simplicidade Textual", "short_sentence_ratio", "Proporção de Sentenças Curtas em relação a todas as sentenças do texto"},
	"simple_word_ratio":                   {"AIC", "Palavra", "Simplicidade Textual", "simple_word_ratio", "Proporção de palavras de conteúdo simples em relação a todas palavras de conteúdo do texto"},
	"subjunctive_future_ratio":            {"GTN", "Sentença", "Léxico Temporal", "subjunctive_future_ratio", "Proporção de Verbos no Futuro do Subjuntivo em relação à quantidade de verbos flexionados no texto"},
	"subjunctive_imperfect_ratio":         {"GTN", "Sentença", "Léxico Temporal", "subjunctive_imperfect_ratio", "Proporção de Verbos no Pretérito Imperfeito do Subjuntivo em relação à quantidade de verbos flexionados no texto"},
	"subjunctive_present_ratio":           {"GTN", "Sentença", "Léxico Temporal", "subjunctive_present_ratio", "Proporção de Verbos no Presente do Subjuntivo em relação à quantidade de verbos flexionados no texto"},
}

type MetrixResultItem struct {
	Index       int    `json:"i"`
	Metric      string `json:"k"`
	Class       string `json:"c"`
	Source      string `json:"s"`
	Level       string `json:"l"`
	Value       string `json:"v"`
	Description string `json:"d"`
}

type MetrixResultOrder []MetrixResultItem

func (a MetrixResultOrder) Len() int      { return len(a) }
func (a MetrixResultOrder) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a MetrixResultOrder) Less(i, j int) bool {
	if a[i].Class == a[j].Class {
		return a[i].Metric < a[j].Metric
	}
	return a[i].Class < a[j].Class
}

type PageInfo struct {
	Version             string                      `json:"version"`
	StaticHash          string                      `json:"shash"`
	MetricList          []MetrixResultItem          `json:"metric_list"`
	SimpligoRankingList []SimpligoRankingResultItem `json:"simpligo_list"`
	ShowResults         bool                        `json:"show_res"`
	Text                string                      `json:"text"`
	Message             string                      `json:"msg"`
	ShowMessage         bool                        `json:"show_msg"`
}

var pageInfo PageInfo

func initialize() {

	pageInfo = PageInfo{}
	pageInfo.Version = "0.0.1"
	pageInfo.StaticHash = "007"

}

func finalize() {
}

func Router() *mux.Router {
	r := mux.NewRouter()
	r.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("./static/"))))
	r.HandleFunc("/", indexHandler)
	r.HandleFunc("/metrixdoc", metrixDocHandler)
	r.HandleFunc("/cohmetrixport", cohMetrixPortHandler)
	r.HandleFunc("/nilcmetrix", nilcMetrixHandler)
	r.HandleFunc("/nilcmetrix-en", nilcMetrixEnHandler)
	r.HandleFunc("/simpligo-ranking", simpligoRankingHandler)
	r.HandleFunc("/api/v1/metrix/{subset}/{key}", metricsHandler).Methods("POST")
	r.HandleFunc("/api/v1/palavras/{retType}/{key}", palavrasHandler)

	return r
}

func main() {

	initialize()

	var httpSrv *http.Server
	httpSrv = makeHTTPServer()

	httpSrv.Addr = ":8080"
	fmt.Printf("Starting HTTP server on %s\n", httpSrv.Addr)
	err := httpSrv.ListenAndServe()
	if err != nil {
		log.Fatalf("httpSrv.ListenAndServe() failed with %s", err)
	}

	defer finalize()

}

func makeHTTPServer() *http.Server {
	mux := Router()
	return &http.Server{
		ReadTimeout:  15 * time.Minute,
		WriteTimeout: 15 * time.Minute,
		IdleTimeout:  15 * time.Minute,
		Handler:      mux,
	}
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	templateHandler(w, r, "index", pageInfo)
}

func metrixDocHandler(w http.ResponseWriter, r *http.Request) {
	templateHandler(w, r, "metrixdoc", pageInfo)
}

func cohMetrixPortHandler(w http.ResponseWriter, r *http.Request) {

	pInfo := pageInfo
	pInfo.ShowResults = false
	pInfo.Message = ""
	pInfo.ShowMessage = false

	text := r.FormValue("text")

	if text != "" {

		nWords := strings.Count(text, " ") + 1
		if nWords > 4000 {
			ret := "Text is too big."
			log.Println(ret)
			pInfo.Message = ret
			pInfo.ShowMessage = true

		} else {

			_, list, err := callMetrix("port", text)
			if err != nil {
				log.Println(err)
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprint(w, "Error "+err.Error())
				return
			}

			sort.Sort(MetrixResultOrder(list))

			for i, it := range list {
				if it.Source == "CMP" || it.Source == "CMP2" {
					it.Index = i + 1
					pInfo.MetricList = append(pInfo.MetricList, it)
				}
			}

			pInfo.ShowResults = true
			pInfo.Text = text
		}
	}

	templateHandler(w, r, "cohmetrixport", pInfo)
}

func nilcMetrixHandler(w http.ResponseWriter, r *http.Request) {
	nilcMetrixCall(w, r, "")
}

func nilcMetrixEnHandler(w http.ResponseWriter, r *http.Request) {
	nilcMetrixCall(w, r, "-en")
}

const SecretKey = "yyy"
const VerifyURL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

type TurnstileResponse struct {
	Success     bool     `json:"success"`
	ErrorCodes  []string `json:"error-codes"`
	ChallengeTS string   `json:"challenge_ts"`
	Hostname    string   `json:"hostname"`
}

type TurnstileRequest struct {
	Secret   string `json:"secret"`
	Response string `json:"response"`
	RemoteIP string `json:"remoteip,omitempty"`
}

func ValidateTurnstile(token string, remoteip string) bool {
	reqBody := TurnstileRequest{
		Secret:   SecretKey,
		Response: token,
		RemoteIP: remoteip,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		log.Println(err)
		return false
	}

	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Post(VerifyURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("Turnstile validation error: %v\n", err)
		return false
	}
	defer resp.Body.Close()

	var result TurnstileResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Println(err)
		return false
	}

	return result.Success
}

func nilcMetrixCall(w http.ResponseWriter, r *http.Request, lang string) {

	pInfo := pageInfo
	pInfo.ShowResults = false

	text := r.FormValue("text")

	if text != "" {
		nWords := strings.Count(text, " ") + 1
		if nWords > 4000 {
			ret := "Text is too big."
			log.Println(ret)
			pInfo.Message = ret
			pInfo.ShowMessage = true

		} else {

			captchaData := r.FormValue("g-recaptcha-response")
			isHuman := ValidateTurnstile(captchaData, r.RemoteAddr)
			if !isHuman {
				ret := "Invalid Re-Captcha Validation."
				log.Println(ret)
				pInfo.Message = ret
				pInfo.ShowMessage = true
			} else {

				_, list, err := callMetrix("_all", text)
				if err != nil {
					log.Println(err)
					w.WriteHeader(http.StatusInternalServerError)
					fmt.Fprint(w, "Error "+err.Error())
					return
				}

				sort.Sort(MetrixResultOrder(list))

				for i, it := range list {
					it.Index = i + 1
					pInfo.MetricList = append(pInfo.MetricList, it)
				}

				pInfo.ShowResults = true
				pInfo.Text = text
			}
		}
	}

	templateHandler(w, r, "nilcmetrix"+lang, pInfo)
}

func templateHandler(w http.ResponseWriter, r *http.Request, pageName string, pInfo PageInfo) {
	t, err := template.New(pageName+".html").Delims("[[", "]]").ParseFiles("./templates/" + pageName + ".html")
	if err != nil {
		fmt.Fprintf(w, "Error openning template: %v", err)
	}

	err = t.Execute(w, pInfo)
	if err != nil {
		fmt.Fprintf(w, "Error parsing template: %v.", err)
	}

}

func metricsHandler(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	subset := vars["subset"]

	if subset == "all" {
		subset = "_all"
	}

	key := vars["key"]

	if key != "yyy" {
		w.WriteHeader(http.StatusForbidden)
		return
	}

	retFormat := r.FormValue("format")
	if retFormat == "plain" {
		w.Header().Set("Content-Type", "text")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}

	ret := ""

	defer r.Body.Close()
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ret += "Error reading req: %v." + err.Error()
		log.Println(ret)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, ret)
		return
	}

	text := string(body)
	ret += text + "\n"
	ret += "-------------------------------" + "\n"

	// nWords := strings.Count(text, " ") + 1
	// if nWords > 4000 {
	// 	ret = "Text is too big."
	// 	log.Println(ret)
	// 	w.WriteHeader(http.StatusInternalServerError)
	// 	fmt.Fprint(w, ret)
	// 	return
	// }

	plain, list, err := callMetrix(subset, text)
	if err != nil {
		ret += "Error " + err.Error()
		log.Println(ret)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, ret)
		return
	}

	if retFormat == "plain" {
		ret += plain
	} else {
		ret = metrixResultToJSON(list)
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(ret))
}

func metrixResultToJSON(list []MetrixResultItem) string {
	ret := "{"
	for _, item := range list {
		ret += "\"" + item.Metric + "\":" + item.Value + ","
	}
	ret = strings.TrimSuffix(ret, ",")
	ret += "}"
	return ret
}

func callMetrix(subset string, text string) (string, []MetrixResultItem, error) {
	shellOut, err := execShellMetrix(subset, text)
	if err != nil {
		return "", []MetrixResultItem{}, err
	}

	list := shellOutToList(shellOut)

	return shellOut, list, nil
}

func preProc(text string) string {
	text = strings.Replace(text, "\"", "{{quotes}}", -1)
	text = strings.Replace(text, "“", "{{quotes}}", -1)
	text = strings.Replace(text, "”", "{{quotes}}", -1)
	text = strings.Replace(text, "\r\n", "{{enter}}", -1)
	text = strings.Replace(text, "\n", "{{enter}}", -1)
	text = strings.Replace(text, "!", "{{exclamation}}", -1)
	text = strings.Replace(text, "#", "{{sharp}}", -1)
	text = strings.Replace(text, "&", "{{ampersand}}", -1)
	text = strings.Replace(text, "%", "{{percent}}", -1)
	text = strings.Replace(text, "$", "{{dollar}}", -1)
	text = strings.Replace(text, "è", "e", -1)
	text = strings.Replace(text, "ì", "i", -1)
	text = strings.Replace(text, "ò", "o", -1)
	text = strings.Replace(text, "ù", "u", -1)
	text = strings.Replace(text, "`", "\"", -1)
	text = strings.Replace(text, "´", "\"", -1)

	text = strings.Replace(text, " à ", "{{crase}}", -1)
	text = strings.Replace(text, "à", "a", -1)
	text = strings.Replace(text, "{{crase}}", " à ", -1)
	return text
}

func execShellMetrix(subset string, text string) (string, error) {
	log.Println("/bin/bash", "-c", "python3 /opt/text_metrics/run"+subset+".py \""+text+"\"")

	text = preProc(text)

	cmd := exec.Command("/bin/bash", "-c", "python3 /opt/text_metrics/run"+subset+".py \""+text+"\"")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("cmd.Run() failed with %v", err.Error())
	}
	fmt.Printf("combined out:\n%s\n", string(out))
	return string(out), nil
}

func shellOutToList(shellOut string) []MetrixResultItem {
	list := []MetrixResultItem{}

	out := strings.Split(shellOut, "++")
	if len(out) < 2 {
		return list
	}
	feats := strings.Split(strings.TrimSpace(out[1]), ",")

	for i, feat := range feats {
		kv := strings.Split(feat, ":")
		if len(kv) > 1 {
			metricName := strings.TrimSpace(kv[0])
			if metric, found := metricMap[metricName]; found {
				list = append(list, MetrixResultItem{i, metric.Key, metric.Class, metric.Source, metric.Level, kv[1], metric.Desc})
			} else {
				list = append(list, MetrixResultItem{i, metricName, "", "", "", kv[1], ""})
			}
		}
	}
	return list
}

func palavrasHandler(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	retType := vars["retType"]

	if retType != "tigerxml" && retType != "flat" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	key := vars["key"]
	if key != "yyy" {
		w.WriteHeader(http.StatusForbidden)
		return
	}

	content := r.FormValue("content")
	if content == "" {
		content = r.FormValue("sentence")
	}
	options := r.FormValue("options")

	resp, err := http.PostForm("http://palavras:8080/palavras/"+retType,
		url.Values{"sentence": {content}, "options": {options}})
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		log.Printf("Error: %v\n", err)
		return
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		panic(fmt.Errorf("Error reading response: %v", err))
	}

	bodyString := string(body)
	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "text")
	fmt.Fprint(w, bodyString)

}
