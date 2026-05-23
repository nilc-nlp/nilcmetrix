NILC-Metrix
============
NILC-Metrix gathers the metrics developed over more than a decade in NILC (Interinstitutional Center for Computational Linguistics), starting with Coh-Metrix-Port, (an adaptation of Coh-Metrix to Portuguese). The main purpose of these metrics is to assess cohesion, coherence and text complexity level. The release of March, 2021 makes available 200 metrics, refactored by Sidney Leal (based on source codes from Carolina Scarton, Andre Cunha and Nathan Hartmann), supported by Magali Duran. All advised by Sandra Aluísio.

License
-------
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)

Citing this work:
-----------------

Public access via SharedIt: https://rdcu.be/doPTI

Leal, S.E., Duran, M.S., Scarton, C.E., Hartmann, N.S., Aluísio, S.M. NILC-Metrix: assessing the complexity of written and spoken language in Brazilian Portuguese. Lang Resources & Evaluation (2023). https://doi.org/10.1007/s10579-023-09693-w

````
@article{NILCMetrix2023,
    author = {Sidney Evaldo Leal and Magali Sanchez Duran and Carolina Evaristo Scarton and Nathan Siegle Hartmann and Sandra Maria Aluísio},
    title = {NILC-Metrix: assessing the complexity of written and spoken language in Brazilian Portuguese},
    journal = {Lang Resources & Evaluation},
    year = {2023},
    url = {https://doi.org/10.1007/s10579-023-09693-w}
}
````

Setup with Docker
-----------------
The steps below match the screencasts linked at the bottom of this section. They assume Docker is installed and running.

### 1. Clone the repository

```console
$ git clone https://github.com/sidleal/nilcmetrix.git
$ cd nilcmetrix
```

### 2. Download the tools bundle

Download [nilc-metrix-tools.zip](https://drive.google.com/file/d/1Ondvnz09RWDAX-1u3GIaXuAmkfKbGtqc/view?usp=sharing) (~2 GB) and unzip it at the repository root. The archive extracts to a `tools/` folder, so the layout becomes `nilcmetrix/tools/`.

### 3. Start PostgreSQL and restore the Coh-Metrix database

```console
$ cd tools/postgres
$ docker run --name pgs_cohmetrix \
    -e POSTGRES_USER=cohmetrix \
    -e POSTGRES_PASSWORD=cohmetrix \
    -v .:/shared \
    -d postgres
```

The first run pulls the `postgres` image. Once the container is up, restore the dump from inside it:

```console
$ docker exec -ti pgs_cohmetrix bash
# cd /shared
# ls
cohmetrix_pt_br  postgres.txt
# pg_restore -U cohmetrix -d cohmetrix cohmetrix_pt_br
```

`pg_restore` may print a couple of warnings (e.g. `role "postgres" does not exist`, `schema "public" already exists`) — these are expected and safe to ignore. Exit the container shell when it finishes.

### 4. Build the main image

From the repository root:

```console
$ ./build.sh
```

This builds the `cohmetrix:focal` image using the local `Dockerfile`. From here you can run the full stack via `./run.sh` or use the pre-built Docker Hub image as described in the next section.

### Screencasts

The original walkthroughs (no audio) cover the same steps:

- Part 1 — clone and download dependencies: https://youtu.be/brBfOuTMOM0
- Part 2 — postgres container and database restore: https://youtu.be/PR2dtr_FMBc
- Part 3 — build the main container and test (metrics that don't depend on the Palavras parser): https://youtu.be/GKQH_1jrmEo

Another way to run with docker-hub pre-built image:
---------------------------------------
Follow Part 1 and 2 from videos above, still need to setup the postgres database in its own container.

After that, instead of building the nilcmetrix container, you can just run a pre-built from docker-hub. Run the run-nilcmetrix-min.sh from root folder, it will run and expose port 8080 with a simple API that return the metrics in json format:

```console
$ ./run-nilcmetrix-min.sh 
Unable to find image 'sidleal/nilcmetrix:latest' locally
latest: Pulling from sidleal/nilcmetrix
96d54c3075c9: Already exists 
...
Digest: sha256:c3f6ce285dc16e5b3f5d3b1d21fe5384294112e15623a21addd072eedb5f5172
Status: Downloaded newer image for sidleal/nilcmetrix:latest
4197a0c2e1644f64b503d623b9bdd264b67599f422f0e1125dd25905e3e64534

$ docker ps
CONTAINER ID   IMAGE                       COMMAND                  CREATED          STATUS          PORTS                                       NAMES
4197a0c2e164   sidleal/nilcmetrix:latest   "/bin/sh -c /opt/tex…"   25 seconds ago   Up 23 seconds   0.0.0.0:8080->8080/tcp, :::8080->8080/tcp   nilcmetrix
677d795e7211   postgres                    "docker-entrypoint.s…"   5 months ago     Up 3 hours      5432/tcp                                    pgs_cohmetrix

$ curl -X POST "http://localhost:8080/api/v1/metrix/_min/yyy?format=json" -H 'Content-Type: text' -d 'Aprender a ler é aprender a ser livre.'
{"adjective_ratio":0.125,"adverbs":0.0,"content_words":0.75,"flesch":103.24,"function_words":0.25,"sentences_per_paragraph":1.0,"syllables_per_content_word":1.83333,"words_per_sentence":8.0,"noun_ratio":0.0,"paragraphs":1,"sentences":1,"words":8,"pronoun_ratio":0.0,"verbs":0.625,"logic_operators":0.0,"and_ratio":0.0,"if_ratio":0.0,"or_ratio":0.0,"negation_ratio":0.0,"cw_freq":1153140.66667,"cw_freq_brwac":5.57717,"cw_freq_bra":5.4175,"min_cw_freq":23996.0,"min_cw_freq_brwac":4.911,"min_freq_brwac":4.911,"min_cw_freq_bra":4.724,"min_freq_bra":4.724,"freq_brwac":5.89538,"freq_bra":5.94137,"hypernyms_verbs":0.25,"brunet":4.69839,"honore":270.927,"personal_pronouns":0.0,"ttr":0.75,"conn_ratio":0.0,"add_neg_conn_ratio":0.0,"add_pos_conn_ratio":0.0,"cau_neg_conn_ratio":0.0,"cau_pos_conn_ratio":0.0,"log_neg_conn_ratio":0.0,"log_pos_conn_ratio":0.0,"tmp_neg_conn_ratio":0.0,"tmp_pos_conn_ratio":0.0,"adjectives_ambiguity":17.0,"adverbs_ambiguity":0,"nouns_ambiguity":0,"verbs_ambiguity":6.4,"yngve":1.77778,"frazier":7.0,"dep_distance":10.0,"cross_entropy":0.75706,"content_density":3.0,"adjacent_refs":0,"anaphoric_refs":0,"adj_arg_ovl":0,"arg_ovl":0,"adj_stem_ovl":0,"stem_ovl":0,"adj_cw_ovl":0,"lsa_adj_mean":0,"lsa_adj_std":0,"lsa_all_mean":0,"lsa_all_std":0,"lsa_paragraph_mean":0,"lsa_paragraph_std":0,"lsa_givenness_mean":0,"lsa_givenness_std":0,"lsa_span_mean":0,"lsa_span_std":0,"negative_words":0.0,"positive_words":0.16667,"ratio_function_to_content_words":0.33333}

```
Example with python:
```
import urllib.request
url = 'http://localhost:8080/api/v1/metrix/_min/yyy?format=json'
text = bytearray('Isso é um texto de teste.', encoding='utf-8')
req = urllib.request.Request(url, data=text, headers={'content-type': 'text/plain'})
response = urllib.request.urlopen(req)
print(response.read().decode('utf8'))
```


---

Mais detalhes:
--------------

NILC-Metrix agrupa as métricas desenvolvidas em mais de uma década no NILC, iniciadas com o Coh-Metrix-Port (uma adaptação da ferramenta Coh-Metrix para o Português Brasileiro). O foco principal das métricas é calcular coesão, coerência e nível de complexidade textual.
Essa versão disponibiliza 200 métricas, detalhadas em [http://fw.nilc.icmc.usp.br:23380/metrixdoc](http://fw.nilc.icmc.usp.br:23380/metrixdoc),
refatoradas por Sidney Leal (com base nos códigos de Carolina Scarton, Andre Cunha e Nathan Hartmann), apoiado por Magali Duran. Todos sob a orientação da professora Sandra Aluísio.


Esse projeto é um fork de: [https://github.com/nilc-nlp/coh-metrix-port](https://github.com/nilc-nlp/coh-metrix-port)

Dependências
------------
- [PostgreSQL](https://www.postgresql.org/)
- [Python 3](http://python.org/)
- [nilc-metrix-tools.zip](https://drive.google.com/file/d/1Ondvnz09RWDAX-1u3GIaXuAmkfKbGtqc/view?usp=sharing) (aprox 2gb) descompactada na raiz

Instale a base de dados do *Coh-Metrix-Dementia*, disponível na subpasta tools/postgres do nilc-metrix-tools.zip com o nome cohmetrix_pt_br (aprox. 15mb):

	$ pg_restore -d cohmetrix_pt_br /caminho/para/projeto/cohmetrix_pt_br

Instale os pacotes do python, opcionalmente em um [virtualenv](https://virtualenv.pypa.io/en/stable/):

	$ pip install -r text_metrics/requirements.txt



Configuração inicial
--------------------
- Adicione o caminho absoluto da pasta `text_metrics/` na variável de ambiente
  `PYTHONPATH`.
- Edite a variável `BASE_DIR` no arquivo `text_metrics/config.py` com o caminho
  para a pasta do projeto.


Como usar
---------
```python
import text_metrics

t = text_metrics.Text("insira aqui seu texto")
# ou de um arquivo
t = text_metrics.Text(filepath="caminho para um arquivo com o texto")

valores = text_metrics.all_metrics.values_for_text(t)
```
