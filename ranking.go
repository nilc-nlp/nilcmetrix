package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/sidleal/simpligo-cloze/web/tools/senter"
)

var simpligoRankingURI = "http://simpligo-ranking:8080/api/v1/ranking/ranking3f"

type SimpligoRankingResultItem struct {
	Index    int    `json:"i"`
	Sentence string `json:"s"`
	Value    int    `json:"v"`
	Color    string `json:"c"`
}

func simpligoRankingHandler(w http.ResponseWriter, r *http.Request) {

	env_URI := os.Getenv("SIMPLIGO_RANKING_URI")
	if len(strings.TrimSpace(env_URI)) == 0 {
		simpligoRankingURI = env_URI
	}
	log.Println(simpligoRankingURI)

	pInfo := pageInfo
	pInfo.ShowResults = false

	text := r.FormValue("text")

	if text != "" {

		nWords := strings.Count(text, " ") + 1
		if nWords > 1000 {
			ret := "Text is too big. / Texto muito longo."
			log.Println(ret)
			pInfo.Message = ret
			pInfo.ShowMessage = true

		} else {

			jsonFeatsList := "["
			sentList := []string{}
			parsed := senter.ParseText(text)
			for _, p := range parsed.Paragraphs {
				for _, s := range p.Sentences {
					_, list, err := callMetrix("_all", s.Text)
					if err != nil {
						log.Println(err)
						w.WriteHeader(http.StatusInternalServerError)
						fmt.Fprint(w, "Error "+err.Error())
						return
					}
					jsonFeatsList += metrixResultToJSON(list) + ","
					sentList = append(sentList, s.Text)
				}
			}
			jsonFeatsList = strings.Trim(jsonFeatsList, ",") + "]"

			list := callSimpligoRanking(jsonFeatsList)
			for i, v := range list {
				it := SimpligoRankingResultItem{}
				it.Index = i + 1
				it.Sentence = sentList[i]
				floatValue, _ := strconv.ParseFloat(v, 64)
				it.Value = int(floatValue * 100)

				if it.Value < 26 {
					it.Color = "green"
				} else if it.Value < 51 {
					it.Color = "orange"
				} else if it.Value < 76 {
					it.Color = "darkorange"
				} else {
					it.Color = "red"
				}

				pInfo.SimpligoRankingList = append(pInfo.SimpligoRankingList, it)
			}

			pInfo.ShowResults = true
			pInfo.Text = text
		}
	}

	templateHandler(w, r, "ranking", pInfo)
}

func callSimpligoRanking(featList string) []string {
	ret := []string{}

	resp, err := http.Post(simpligoRankingURI, "text", bytes.NewReader([]byte(featList)))
	if err != nil {
		log.Println("Error: " + err.Error())
		return ret
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Println("Error reading response: " + err.Error())
		return ret
	}

	log.Println(string(body))
	ret = strings.Split(string(body), ",")

	return ret
}
