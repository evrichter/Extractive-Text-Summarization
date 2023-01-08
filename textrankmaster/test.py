from text_rank.text_rank import TextRank



def tokenizer(document):
    """
    """
    text = re.sub('[^A-Za-zÀ-ž\u0370-\u03FF\u0400-\u04FF]', ' ', document)
    tokens = text.lower().split()
    doc = nlp(" ".join(tokens))
    tokens = [x.lemma_ for x in doc]
    return tokens


text = 'Ob die APA die Qualität auch mit weniger Mitarbeitern aufrechterhalten könne?.Hauptkritikpunkt der Belegschaft ist, dass Sparmaßnahmen angekündigt würden, obwohl die APA seit Jahren Gewinne schreibe..Persönlich habe er außerdem nur eine geringe Beteiligung an der Betriebsversammlung wahrgenommen..Die als Genossenschaft organisierte Agentur gehört zu 45 Prozent dem ORF, den Rest teilen sich 13 Tageszeitungen..Für 2016 steige das Personalbudget sogar, allerdings würden automatische Gehaltserhöhungen die Kosten für den einzelnen Mitarbeiter steigen lassen.. Der Betriebsrat glaubt, dass man die Kosten einfach unbegrenzt weiterlaufen lassen kann, so Kropsch.'

tr = TextRank()
sents_tokens = [sent.split(" ") for sent in text.split(".")]
tr.set_sentences(sents_tokens)
import pdb
pdb.set_trace()
print(tr.textrank)
