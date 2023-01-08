Project implementation for extractive summarization using topic modelling, sentence clustering and text rank


Project Structure

.
├── data                   # csv files for documents splitted into train and test
├── output                    # saved lda model and dictionary
├── textrankmaster                  # Source files for text rank algorithm
├── data_analysis          # analyse the total data being used
├── documentSummaries.py                   # extract extractive summary from document
├── instructions.txt
├── requirements.txt               # packages to be installed beforehand
├── rogue_r.py                    # code to calculate rogue
├── run_evaluation.sh                     # script to run evaluation on unseen and seen data
├── run_training.sh                    # script to train lda
├── run_inference.sh                   # script to extract summary for single document
├── topic_modelling                    # code for training lda module
└── README.md

# DATA DOWNLOADING INSTRUCTION #

the training file with 99000 documents is heavy in size (around 500 MB) and hence i uploaded it to the drive with public access where one can easily download. But you can still run the evaluation and inference since the trained model is already inside this directory.

LINK: https://drive.google.com/file/d/1dYuJm92pZRcd8roCQagiLN2TPZccrQpB/view?usp=sharing

# Project Execution Guide #

1. set up the virtual environment

```
virtualenv <env name>
source <env name>/bin/activate
pip install -r requirements.txt



```

1. Training the documents for topic modelling.

```
./run_training.sh

```

The model is already trained and saved inside output folder which could be used for evaluation. For 99000 documents, training took 1 and half hour. Hence, not recommended to train. but of you want to make sure to pass the required training and test files required by the file topic_modelling.py. To see how to pass argument, run:

```
topic_modelling.py -h
```

2. evaluating on unseen and seen documents.

100 seen and unseen data are available in the data folder, Evaluation result can be calculated by running:

```
./run_evaluation.sh

```

--------------------------------------------------------------------------Result---------------------------------------------------------------------------------------------------------------------------
TEST FOR UNSEEN DOCUMENT
Evaluation with Avg
	rouge-1:	P: 20.82	R: 24.63	F1: 20.73
	rouge-2:	P:  3.97	R:  4.50	F1:  3.89
	rouge-3:	P:  1.16	R:  1.42	F1:  1.17
	rouge-4:	P:  0.37	R:  0.46	F1:  0.39
	rouge-l:	P: 23.57	R: 27.47	F1: 23.87
	rouge-w:	P: 13.02	R:  9.00	F1:  9.70

Evaluation with Best
	rouge-1:	P: 20.82	R: 24.63	F1: 20.73
	rouge-2:	P:  3.97	R:  4.50	F1:  3.89
	rouge-3:	P:  1.16	R:  1.42	F1:  1.17
	rouge-4:	P:  0.37	R:  0.46	F1:  0.39
	rouge-l:	P: 23.57	R: 27.47	F1: 23.87
	rouge-w:	P: 13.02	R:  9.00	F1:  9.70


TEST FOR SEEN DOCUMENT
Evaluation with Avg
	rouge-1:	P: 16.99	R: 31.70	F1: 20.08
	rouge-2:	P:  3.59	R:  7.60	F1:  4.40
	rouge-3:	P:  0.94	R:  2.40	F1:  1.25
	rouge-4:	P:  0.34	R:  0.98	F1:  0.48
	rouge-l:	P: 20.01	R: 33.99	F1: 23.48
	rouge-w:	P: 11.32	R: 13.12	F1: 10.74

Evaluation with Best
	rouge-1:	P: 16.99	R: 31.70	F1: 20.08
	rouge-2:	P:  3.59	R:  7.60	F1:  4.40
	rouge-3:	P:  0.94	R:  2.40	F1:  1.25
	rouge-4:	P:  0.34	R:  0.98	F1:  0.48
	rouge-l:	P: 20.01	R: 33.99	F1: 23.48
	rouge-w:	P: 11.32	R: 13.12	F1: 10.74



3. Similarly, to run the inference
You can pass your own document, which should be a german article inside the file data/text_inference.txt. But there is already a document available inside the text file. To print its summary, run:

```
./run_inference.sh
```

-----------------------------------------------------------------------------------------RESULT------------------------------------------------------------------------------------------------------------

document to be summarized:
Nach dem Abitur an der Gisela-Oberrealschule in München-Schwabing studierte Hofmann, dessen Vater Regierungsvizepräsident in Ansbach war, Rechts- und Staatswissenschaften an der Ludwig-Maximilians-Universität München sowie der Friedrich-Alexander-Universität Erlangen-Nürnberg. Er war danach von 1958 bis 1961 als Regierungsassessor und zuletzt als Regierungsrat in Augsburg und Schwabmünchen innerhalb der bayerischen Innenverwaltung tätig. Während seines Studiums in Erlangen wurde er Mitglied der Studentenverbindung "Corps Bavaria Erlangen". 1961 wurde Hofmann Persönlicher Referent von Walter Scheel, der kurz zuvor zum ersten Bundesminister für wirtschaftliche Zusammenarbeit ernannt worden war. Nach vierjähriger Tätigkeit wurde er 1965 in diesem Bundesministerium Referatsleiter für Südostasien und Ostasien, ehe er vom 1. Januar 1969 an Leiter des dortigen Referats für internationale Fragen der Entwicklungspolitik war. Nachdem Walter Scheel Bundesaussenminister der sozialliberalen Koalition geworden war, ernannte er Hofmann im Oktober 1969 zum Leiter des Ministerbüros des Auswärtigen Amtes. Im September 1972 folgte die Beförderung zum Ministerialdirigenten und Ernennung zum Leiter des Leitungsstabes des Auswärtigen Amtes. Im Anschluss daran wurde Hofmann im Juli 1973 vom Bundesvorstand der FDP zum Bundesgeschäftsführer der FDP berufen und übernahm dieses Amt am 1. September 1973 als Nachfolger von Joachim Stancke. Während dieser Zeit war er zugleich auch Geschäftsführer Inland der Friedrich-Naumann-Stiftung. Hofmann ist Mitglied der Jury des Verbandes Liberaler Akademiker zur Vergabe des Arno-Esch-Preises. 1977 kehrte er ins Auswärtige Amt zurück und wurde als Nachfolger von Werner Ahrens zum Botschafter in Dänemark ernannt und behielt dieses Amt bis zu seiner Ablösung durch Rudolf Jestaedt 1981. Während dieser Zeit war er unter anderem 1980 auch Stellvertreter des Parlamentarischen Staatssekretärs im Bundesministerium für Jugend, Familie und Gesundheit, Fred Zander, als Leiter der Delegation beim Frauenkongress der Vereinten Nationen in Kopenhagen. Danach wurde Hofmann 1981 Nachfolger von Helmut Redies als Botschafter in Venezuela und bekleidete diese Funktion bis zu seiner Ablösung durch Hans Werner Loeck 1985. Danach erhielt er seine Akkreditierung als Botschafter in Norwegen, wo er Nachfolger des im Amt verstorbenen Johannes Balser wurde. Als solcher hielt Hofmann auch Vorträge vor der Deutsch-Norwegischen Gesellschaft zu bilateralen diplomatischen Beziehungen. Nach siebenjähriger Verwendung in Norwegen folgte ihm 1992 Helmut Wegner, während er wiederum Nachfolger von Reinhold Schenk als Botschafter in Schweden wurde. Das Amt des Botschafters in Schweden übte Hofmann bis zu seiner Versetzung in den Ruhestand 1997 aus.

Predicted Summary is:
 Danach erhielt er seine Akkreditierung als Botschafter in Norwegen, wo er Nachfolger des im Amt verstorbenen Johannes Balser wurde
 1961 wurde Hofmann Persönlicher Referent von Walter Scheel, der kurz zuvor zum ersten Bundesminister für wirtschaftliche Zusammenarbeit ernannt worden war
 Hofmann ist Mitglied der Jury des Verbandes Liberaler Akademiker zur Vergabe des Arno-Esch-Preises

