
import os
import pandas

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import requests as r


#-------------------------------------------------------------------------
# trier la liste sur les valeurs et non sur la clef
#-------------------------------------------------------------------------
if 1 :
    
    pays_lang_list = pd.DataFrame()  
    pays_lang_list.fr = "AD=ANDORRE;AE=EMIRATS ARABES UNIS;AF=AFGHANISTAN;AG=ANTIGUA-ET-BARBUDA;AI=ANGUILLA;AL=ALBANIE;AM=ARMENIE;AN=ANTILLES NEERLANDAISES;AO=ANGOLA;AQ=ANTARCTIQUE;AR=ARGENTINE;AS=SAMOA AMERICAINES;AT=AUTRICHE;AU=AUSTRALIE;AW=ARUBA;AX=ALAND, ILES;AZ=AZERBAIDJAN;BA=BOSNIE-HERZEGOVINE;BB=BARBADE;BD=BANGLADESH;BE=BELGIQUE;BF=BURKINA FASO;BG=BULGARIE;BH=BAHREIN;BI=BURUNDI;BJ=BENIN;BL=SAINT-BARTHELEMY;BM=BERMUDES;BN=BRUNEI DARUSSALAM;BO=BOLIVIE;BQ=BONAIRE, SAINT-EUSTACHE ET SABA;BR=BRESIL;BS=BAHAMAS;BT=BHOUTAN;BV=BOUVET, ILE;BW=BOTSWANA;BY=BELARUS;BZ=BELIZE;CA=CANADA;CC=COCOS (KEELING), ILES;CD=CONGO, LA REP. DEMOCRATIQUE DU;CF=CENTRAFRICAINE, REPUBLIQUE;CG=CONGO;CH=SUISSE;CI=COTE D'IVOIRE;CK=COOK, ILES;CL=CHILI;CM=CAMEROUN;CN=CHINE;CO=COLOMBIE;CR=COSTA RICA;CS=SERBIE MONTENEGRO;CU=CUBA;CV=CABO VERDE;CW=CURAÇAO;CX=CHRISTMAS, ILE;CY=CHYPRE;CZ=TCHEQUE, REPUBLIQUE;DE=ALLEMAGNE;DJ=DJIBOUTI;DK=DANEMARK;DM=DOMINIQUE;DO=DOMINICAINE, REPUBLIQUE;DZ=ALGERIE;EC=EQUATEUR;EE=ESTONIE;EG=EGYPTE;EH=SAHARA OCCIDENTAL;ER=ERYTHREE;ES=ESPAGNE;ET=ETHIOPIE;FI=FINLANDE;FJ=FIDJI;FK=FALKLAND, ILES (MALVINAS);FM=MICRONESIE, ETATS FEDERES DE;FO=FEROE, ILES;FR=FRANCE;GA=GABON;GB=ROYAUME-UNI;GD=GRENADE;GE=GEORGIE;GF=GUYANE FRANCAISE;GG=GUERNESEY;GH=GHANA;GI=GIBRALTAR;GL=GROENLAND;GM=GAMBIE;GN=GUINEE;GP=GUADELOUPE;GQ=GUINEE EQUATORIALE;GR=GRECE;GS=GEORGIE DU SUD-ET-LES ILES SANDWICH DU SUD;GT=GUATEMALA;GU=GUAM;GW=GUINEE-BISSAU;GY=GUYANA;HK=HONG KONG;HM=HEARD-ET-ILES MACDONALD, ILE;HN=HONDURAS;HR=CROATIE;HT=HAITI;HU=HONGRIE;ID=INDONESIE;IE=IRLANDE;IL=ISRAEL;IM=ILE DE MAN;IN=INDE;IO=OCEAN INDIEN, TERRITOIRE BRITANNIQUE DE L';IQ=IRAQ;IR=IRAN, REPUBLIQUE ISLAMIQUE D';IS=ISLANDE;IT=ITALIE;JE=JERSEY;JM=JAMAIQUE;JO=JORDANIE;JP=JAPON;KE=KENYA;KG=KIRGHIZISTAN;KH=CAMBODGE;KI=KIRIBATI;KM=COMORES;KN=ST.KITTS-ET-NEVIS;KP=COREE, REP.POP.DEM.;KR=COREE, REPUBLIQUE DE;KW=KOWEIT;KY=CAIMANS, ILES;KZ=KAZAKHSTAN;LA=LAO, REPUBLIQUE DEMOCRATIQUE POPULAIRE;LB=LIBAN;LC=STE. LUCIE;LI=LIECHTENSTEIN;LK=SRI LANKA;LR=LIBERIA;LS=LESOTHO;LT=LITUANIE;LU=LUXEMBOURG;LV=LETTONIE;LY=LIBYE;MA=MAROC;MC=MONACO;MD=MOLDOVA, REPUBLIQUE DE;ME=MONTENEGRO;MF=SAINT-MARTIN(PARTIE FRANCAISE);MG=MADAGASCAR;MH=MARSHALL, ILES;MK=MACEDOINE;ML=MALI;MM=MYANMAR;MN=MONGOLIE;MO=MACAO;MP=MARIANNES DU NORD, ILES;MQ=MARTINIQUE;MR=MAURITANIE;MS=MONTSERRAT;MT=MALTE;MU=MAURICE;MV=MALDIVES;MW=MALAWI;MX=MEXIQUE;MY=MALAISIE;MZ=MOZAMBIQUE;NA=NAMIBIE;NC=NOUVELLE-CALEDONIE;NE=NIGER;NF=NORFOLK, ILE;NG=NIGERIA;NI=NICARAGUA;NL=PAYS-BAS;NO=NORVEGE;NP=NEPAL;NR=NAURU;NU=NIUE;NZ=NOUVELLE-ZELANDE;OM=OMAN;PA=PANAMA;PE=PEROU;PF=POLYNESIE FRANCAISE;PG=PAPOUASIE-NOUVELLE-GUINEE;PH=PHILIPPINES;PK=PAKISTAN;PL=POLOGNE;PM=SAINT-PIERRE-ET-MIQUELON;PN=PITCAIRN;PR=PORTO RICO;PS=PALESTINE, ÉTAT DE;PT=PORTUGAL;PW=PALAOS;PY=PARAGUAY;QA=QATAR;RE=REUNION;RO=ROUMANIE;RS=SERBIE;RU=RUSSIE, FEDERATION DE;RW=RWANDA;SA=ARABIE SAOUDITE;SB=SALOMON, ILES;SC=SEYCHELLES;SD=SOUDAN;SE=SUEDE;SG=SINGAPOUR;SH=SAINTE-HELENE;SI=SLOVENIE;SJ=SVALBARD ET ILE JAN MAYEN;SK=SLOVAQUIE;SL=SIERRA LEONE;SM=SAINT-MARIN;SN=SENEGAL;SO=SOMALIE;SR=SURINAME;SS=SOUDAN DU SUD;ST=SAO TOME-ET-PRINCIPE;SV=EL SALVADOR;SX=SAINT-MARTIN (PARTIE NÉERLANDAISE);SY=RÉPUBLIQUE ARABE SYRIENNE;SZ=SWAZILAND;TC=TURKS-ET-CAICOS, ILES;TD=TCHAD;TF=TERRES AUSTRALES FRANCAISES;TG=TOGO;TH=THAILANDE;TJ=TADJIKISTAN;TK=TOKELAU;TL=TIMOR-LESTE;TM=TURKMENISTAN;TN=TUNISIE;TO=TONGA;TP=TIMOR ORIENTAL;TR=TURQUIE;TT=TRINITE-ET-TOBAGO;TV=TUVALU;TW=TAIWAN, PROVINCE DE CHINE;TZ=TANZANIE, REP. UNIE DE;UA=UKRAINE;UG=OUGANDA;UM=ILES MINEURES ELOIGNEES DES ETATS-UNIS;US=ÉTATS-UNIS D'AMÉRIQUE;UY=URUGUAY;UZ=OUZBEKISTAN;VA=CITE DU VATICAN;VC=ST. VINCENT-ET-LES-GRANADINES;VE=VENEZUELA;VG=ILES VIERGES BRITANNIQUES;VI=ILES VIERGES DES ETATS-UNIS;VN=VIET NAM;VU=VANUATU;WF=WALLIS ET FUTUNA;WS=SAMOA;XK=KOSOVO;YE=YEMEN;YT=MAYOTTE;YU=YOUGOSLAVIE;ZA=AFRIQUE DU SUD;ZM=ZAMBIE;ZW=ZIMBABWE"
    pays_lang_list.de = "AD=ANDORRA;AE=VEREINIGTE ARABISCHE EMIRATE;AF=AFGHANISTAN;AG=ANTIGUA UND BARBUDA;AI=ANGUILLA;AL=ALBANIEN;AM=ARMENIEN;AN=NIEDERLÄNDISCHE ANTILLEN;AO=ANGOLA;AQ=ANTARKTIKA;AR=ARGENTINIEN;AS=AMERIKANISCH SAMOA;AT=ÖSTERREICH;AU=AUSTRALIEN;AW=ARUBA;AX=ALAND INSELN;AZ=ASERBAIDSCHAN;BA=BOSNIEN UND HERZEGOWINA;BB=BARBADOS;BD=BANGLADESCH;BE=BELGIEN;BF=BURKINA FASO;BG=BULGARIEN;BH=BAHRAIN;BI=BURUNDI;BJ=BENIN;BL=SAINT BARTHELEMY;BM=BERMUDA;BN=BRUNEI DARUSSALAM;BO=BOLIVIEN;BQ=BONAIRE, SINT EUSTATIUS UND SABA;BR=BRASILIEN;BS=BAHAMAS;BT=BHUTAN;BV=BOUVETINSEL;BW=BOTSWANA;BY=BELARUS (WEIßRUSSLAND);BZ=BELIZE;CA=KANADA;CC=KOKOSINSELN;CD=KONGO, DEMOKRATISCHE REPUBLIK;CF=ZENTRALAFRIKANISCHE REPUBLIK;CG=KONGO;CH=SCHWEIZ;CI=COTE D'IVOIRE (ELFENBEINKÜSTE);CK=COOKINSELN;CL=CHILE;CM=KAMERUN;CN=CHINA, VOLKSREPUBLIK;CO=KOLUMBIEN;CR=COSTA RICA;CS=SERBIEN MONTENEGRO;CU=KUBA;CV=KAP VERDE;CW=CURAÇAO;CX=WEIHNACHTSINSEL;CY=ZYPERN;CZ=TSCHECHISCHE REPUBLIK;DE=DEUTSCHLAND;DJ=DSCHIBOUTI;DK=DÄNEMARK;DM=DOMINICA;DO=DOMINIKANISCHE REPUBLIK;DZ=ALGERIEN;EC=ECUADOR;EE=ESTLAND;EG=ÄGYPTEN;EH=WESTSAHARA;ER=ERITREA;ES=SPANIEN;ET=ÄTHIOPIEN;FI=FINNLAND;FJ=FIDSCHI;FK=FALKLANDINSELN;FM=MIKRONESIEN;FO=FÄRÖER;FR=FRANKREICH;GA=GABUN;GB=GROßBRITANNIEN;GD=GRENADA;GE=GEORGIEN;GF=FRANZÖSISCH-GUAYANA;GG=GUERNSEY (KANALINSEL);GH=GHANA;GI=GIBRALTAR;GL=GRÖNLAND;GM=GAMBIA;GN=GUINEA;GP=GUADELOUPE;GQ=ÄQUATORIAL-GUINEA;GR=GRIECHENLAND;GS=SÜDGEORGIEN UND DIE SÜDLICHEN SANDWICHINSELN;GT=GUATEMALA;GU=GUAM;GW=GUINEA-BISSAU;GY=GUYANA;HK=HONGKONG;HM=HERALD UND MCDONALDINSELN;HN=HONDURAS;HR=KROATIEN;HT=HAITI;HU=UNGARN;ID=INDONESIEN;IE=IRLAND;IL=ISRAEL;IM=INSEL MAN;IN=INDIEN;IO=BRIT. TERRITORIUM IM INDISCHEN OZEAN;IQ=IRAK;IR=IRAN, ISLAMISCHE REPUBLIK;IS=ISLAND;IT=ITALIEN;JE=JERSEY (KANALINSEL);JM=JAMAIKA;JO=JORDANIEN;JP=JAPAN;KE=KENIA;KG=KIRGISISTAN;KH=KAMBODSCHA;KI=KIRIBATI;KM=KOMOREN;KN=ST. KITTS UND NEVIS;KP=KOREA, DEMOKR. VOLKSREPUBLIK (NORDKOREA);KR=KOREA, REPUBLIK (SÜDKOREA);KW=KUWAIT;KY=KAIMANINSELN;KZ=KASACHSTAN;LA=LAOS, DEMOKRATISCHE VOLKSREPUBLIK;LB=LIBANON;LC=ST. LUCIA;LI=LIECHTENSTEIN;LK=SRI LANKA;LR=LIBERIA;LS=LESOTHO;LT=LITAUEN;LU=LUXEMBURG;LV=LETTLAND;LY=LIBYEN;MA=MAROKKO;MC=MONACO;MD=MOLDAWIEN;ME=MONTENEGRO;MF=SAINT MARTIN;MG=MADAGASKAR;MH=MARSHALLINSELN;MK=MAZEDONIEN;ML=MALI;MM=MYANMAR (BURMA);MN=MONGOLEI;MO=MACAU;MP=NÖRDLICHE MARIANEN;MQ=MARTINIQUE;MR=MAURETANIEN;MS=MONTSERRAT;MT=MALTA;MU=MAURITIUS;MV=MALEDIVEN;MW=MALAWI;MX=MEXIKO;MY=MALAYSIA;MZ=MOSAMBIK;NA=NAMIBIA;NC=NEUKALEDONIEN;NE=NIGER;NF=NORFOLKINSEL;NG=NIGERIA;NI=NICARAGUA;NL=NIEDERLANDE;NO=NORWEGEN;NP=NEPAL;NR=NAURU;NU=NIUE;NZ=NEUSEELAND;OM=OMAN;PA=PANAMA;PE=PERU;PF=FRANZÖSISCH-POLYNESIEN;PG=PAPUA NEUGUINEA;PH=PHILIPPINEN;PK=PAKISTAN;PL=POLEN;PM=ST. PIERRE UND MIQUELON;PN=PITCAIRNINSELN;PR=PUERTO RICO;PS=STAAT PALÄSTINA;PT=PORTUGAL;PW=PALAU;PY=PARAGUAY;QA=KATAR;RE=RÉUNION;RO=RUMÄNIEN;RS=SERBIEN;RU=RUSSISCHE FÖDERATION;RW=RUANDA;SA=SAUDI-ARABIEN;SB=SALOMON-INSELN;SC=SEYCHELLEN;SD=SUDAN;SE=SCHWEDEN;SG=SINGAPUR;SH=ST. HELENA;SI=SLOWENIEN;SJ=SVALBARD UND JAN MAYEN;SK=SLOWAKEI;SL=SIERRA LEONE;SM=SAN MARINO;SN=SENEGAL;SO=SOMALIA;SR=SURINAME;SS=SUDSUDAN;ST=SAO-TOMÉ UND PRINCIPE;SV=EL SALVADOR;SX=SINT MAARTEN (HOLLÄNDISCHER TEIL);SY=SYRIEN, ARABISCHE REPUBLIK;SZ=SWASILAND;TC=TURKS- UND CAICOSINSELN;TD=TSCHAD;TF=FRANZÖSISCHE SÜD- UND ANTARKTISGEBIETE;TG=TOGO;TH=THAILAND;TJ=TADSCHIKISTAN;TK=TOKELAU;TL=OSTTIMOR;TM=TURKMENISTAN;TN=TUNESIEN;TO=TONGA;TP=OSTTIMOR;TR=TÜRKEI;TT=TRINIDAD UND TOBAGO;TV=TUVALU;TW=TAIWAN;TZ=TANSANIA, VEREINIGTE REPUBLIK;UA=UKRAINE;UG=UGANDA;UM=UNITED STATES MINOR OUTLYING ISLANDS;US=VEREINIGTE STAATEN VON AMERIKA;UY=URUGUAY;UZ=USBEKISTAN;VA=VATIKANSTADT;VC=ST. VINCENT UND DIE GRENADINEN;VE=VENEZUELA;VG=BRITISCHE JUNGFERNINSELN;VI=AMERIKANISCHE JUNGFERNINSELN;VN=VIETNAM;VU=VANUATU;WF=WALLIS UND FUTUNA;WS=SAMOA;XK=KOSOVO;YE=JEMEN;YT=MAYOTTE;YU=JUGOSLAVIEN;ZA=SÜDAFRIKA;ZM=SAMBIA;ZW=SIMBABWE"
    pays_lang_list.en = "AD=ANDORRA;AE=UNITED ARAB EMIRATES;AF=AFGHANISTAN;AG=ANTIGUA AND BARBUDA;AI=ANGUILLA;AL=ALBANIA;AM=ARMENIA;AN=NETHERLANDS ANTILLES;AO=ANGOLA;AQ=ANTARCTICA;AR=ARGENTINA;AS=AMERICAN SAMOA;AT=AUSTRIA;AU=AUSTRALIA;AW=ARUBA;AX=ALAND ISLANDS;AZ=AZERBAIJAN;BA=BOSNIA AND HERZEGOVINA;BB=BARBADOS;BD=BANGLADESH;BE=BELGIUM;BF=BURKINA FASO;BG=BULGARIA;BH=BAHRAIN;BI=BURUNDI;BJ=BENIN;BL=SAINT BARTHELEMY;BM=BERMUDA;BN=BRUNEI DARUSSALAM;BO=BOLIVIA;BQ=BONAIRE, SINT EUSTATIUS AND SABA;BR=BRAZIL;BS=BAHAMAS;BT=BHUTAN;BV=BOUVET ISLAND;BW=BOTSWANA;BY=BELARUS;BZ=BELIZE;CA=CANADA;CC=COCOS (KEELING) ISLANDS;CD=CONGO, THE DEMOCRATIC REPUBLIC OF THE;CF=CENTRAL AFRICAN REPUBLIC;CG=CONGO;CH=SWITZERLAND;CI=COTE D'IVOIRE;CK=COOK ISLANDS;CL=CHILE;CM=CAMEROON;CN=CHINA;CO=COLOMBIA;CR=COSTA RICA;CS=SERBIA MONTENEGRO;CU=CUBA;CV=CABO VERDE;CW=CURAÇAO;CX=CHRISTMAS ISLAND;CY=CYPRUS;CZ=CZECH REPUBLIC;DE=GERMANY;DJ=DJIBOUTI;DK=DENMARK;DM=DOMINICA;DO=DOMINICAN REPUBLIC;DZ=ALGERIA;EC=ECUADOR;EE=ESTONIA;EG=EGYPT;EH=WESTERN SAHARA;ER=ERITREA;ES=SPAIN;ET=ETHIOPIA;FI=FINLAND;FJ=FIJI;FK=FALKLAND ISLANDS (MALVINAS);FM=MICRONESIA, FEDERATED STATES OF;FO=FAROE ISLANDS;FR=FRANCE;GA=GABON;GB=UNITED KINGDOM;GD=GRENADA;GE=GEORGIA;GF=FRENCH GUIANA;GG=GUERNSEY;GH=GHANA;GI=GIBRALTAR;GL=GREENLAND;GM=GAMBIA;GN=GUINEA;GP=GUADELOUPE;GQ=EQUATORIAL GUINEA;GR=GREECE;GS=SOUTH GEORGIA AND SOUTH SANDWICH ISLANDS;GT=GUATEMALA;GU=GUAM;GW=GUINEA-BISSAU;GY=GUYANA;HK=HONG KONG;HM=HEARD ISLAND AND MCDONALD ISLANDS;HN=HONDURAS;HR=CROATIA;HT=HAITI;HU=HUNGARY;ID=INDONESIA;IE=IRELAND;IL=ISRAEL;IM=ISLE OF MAN;IN=INDIA;IO=BRITISH INDIAN OCEAN TERRITORY;IQ=IRAQ;IR=IRAN, ISLAMIC REPUBLIC OF;IS=ICELAND;IT=ITALY;JE=JERSEY;JM=JAMAICA;JO=JORDAN;JP=JAPAN;KE=KENYA;KG=KYRGYZSTAN;KH=CAMBODIA;KI=KIRIBATI;KM=COMOROS;KN=SAINT KITTS AND NEVIS;KP=KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF;KR=KOREA, REPUBLIC OF;KW=KUWAIT;KY=CAYMAN ISLANDS;KZ=KAZAKHSTAN;LA=LAO PEOPLE'S DEMOCRATIC REPUBLIC;LB=LEBANON;LC=SAINT LUCIA;LI=LIECHTENSTEIN;LK=SRI LANKA;LR=LIBERIA;LS=LESOTHO;LT=LITHUANIA;LU=LUXEMBOURG;LV=LATVIA;LY=LIBYA;MA=MOROCCO;MC=MONACO;MD=MOLDOVA, REPUBLIC OF;ME=MONTENEGRO;MF=SAINT MARTIN (FRENCH PART);MG=MADAGASCAR;MH=MARSHALL ISLANDS;MK=MACEDONIA;ML=MALI;MM=MYANMAR;MN=MONGOLIA;MO=MACAO;MP=NORTHERN MARIANA ISLANDS;MQ=MARTINIQUE;MR=MAURITANIA;MS=MONTSERRAT;MT=MALTA;MU=MAURITIUS;MV=MALDIVES;MW=MALAWI;MX=MEXICO;MY=MALAYSIA;MZ=MOZAMBIQUE;NA=NAMIBIA;NC=NEW CALEDONIA;NE=NIGER;NF=NORFOLK ISLAND;NG=NIGERIA;NI=NICARAGUA;NL=NETHERLANDS;NO=NORWAY;NP=NEPAL;NR=NAURU;NU=NIUE;NZ=NEW ZEALAND;OM=OMAN;PA=PANAMA;PE=PERU;PF=FRENCH POLYNESIA;PG=PAPUA NEW GUINEA;PH=PHILIPPINES;PK=PAKISTAN;PL=POLAND;PM=SAINT PIERRE AND MIQUELON;PN=PITCAIRN;PR=PUERTO RICO;PS=PALESTINE, STATE OF;PT=PORTUGAL;PW=PALAU;PY=PARAGUAY;QA=QATAR;RE=REUNION;RO=ROMANIA;RS=SERBIA;RU=RUSSIAN FEDERATION;RW=RWANDA;SA=SAUDI ARABIA;SB=SOLOMON ISLANDS;SC=SEYCHELLES;SD=SUDAN;SE=SWEDEN;SG=SINGAPORE;SH=SAINT HELENA;SI=SLOVENIA;SJ=SVALBARD AND JAN MAYEN;SK=SLOVAKIA;SL=SIERRA LEONE;SM=SAN MARINO;SN=SENEGAL;SO=SOMALIA;SR=SURINAME;SS=SOUTH SUDAN;ST=SAO TOME AND PRINCIPE;SV=EL SALVADOR;SX=SINT MAARTEN (DUTCH PART);SY=SYRIAN ARAB REPUBLIC;SZ=SWAZILAND;TC=TURKS AND CAICOS ISLANDS;TD=CHAD;TF=FRENCH SOUTHERN TERRITORIES;TG=TOGO;TH=THAILAND;TJ=TAJIKISTAN;TK=TOKELAU;TL=TIMOR-LESTE;TM=TURKMENISTAN;TN=TUNISIA;TO=TONGA;TP=WESTERN TIMOR;TR=TURKEY;TT=TRINIDAD AND TOBAGO;TV=TUVALU;TW=TAIWAN, PROVINCE OF CHINA;TZ=TANZANIA, UNITED REPUBLIC OF;UA=UKRAINE;UG=UGANDA;UM=UNITED STATES MINOR OUTLYING ISLANDS;US=UNITED STATES OF AMERICA (THE);UY=URUGUAY;UZ=UZBEKISTAN;VA=VATICAN CITY STATE;VC=SAINT VINCENT AND THE GRENADINES;VE=VENEZUELA;VG=VIRGIN ISLANDS, BRITISH;VI=VIRGIN ISLANDS, U.S.;VN=VIET NAM;VU=VANUATU;WF=WALLIS AND FUTUNA;WS=SAMOA;XK=KOSOVO;YE=YEMEN;YT=MAYOTTE;YU=YOUGOSLAVIA (FORMER);ZA=SOUTH AFRICA;ZM=ZAMBIA;ZW=ZIMBABWE"
    pays_lang_list.it = "AD=ANDORRA;AE=EMIRATI ARABI UNITI;AF=AFGHANISTAN;AG=ANTIGUA/BARBUDA;AI=ANGUILLA;AL=ALBANIA;AM=ARMENIA;AN=ANTILLE OLANDESI;AO=ANGOLA;AQ=ANTARTIDE;AR=ARGENTINA;AS=SAMOA AMERICANE;AT=AUSTRIA;AU=AUSTRALIA;AW=ARUBA;AX=ISOLE ALAND;AZ=AZERBAIGIAN;BA=BOSNIA ED ERZEGOVINA;BB=BARBADOS;BD=BANGLADESH;BE=BELGIO;BF=BURKINA FASO;BG=BULGARIA;BH=BAHRAIN;BI=BURUNDI;BJ=BENIN;BL=SAINT BARTHELEMY;BM=BERMUDA;BN=BRUNEI;BO=BOLIVIA;BQ=BONAIRE, SINT EUSTATIUS E SABA;BR=BRASILE;BS=BAHAMAS;BT=BHUTAN;BV=ISOLA BOUVET;BW=BOTSWANA;BY=BIELORUSSIA;BZ=BELIZE;CA=CANADA;CC=ISOLE COCOS E KEELING;CD=REPUBBLICA DEMOCRATICA DEL CONGO;CF=REPUBBLICA CENTRAFRICANA;CG=CONGO, REPUBBLICA;CH=SVIZZERA;CI=COSTA D'AVORIO;CK=ISOLE COOK;CL=CILE;CM=CAMERUN;CN=CINA;CO=COLOMBIA;CR=COSTA RICA;CS=SERBIA MONTENEGRO;CU=CUBA;CV=CAPO VERDE;CW=CURAÇAO;CX=ISOLA DEL NATALE;CY=CIPRO;CZ=CECA, REPUBBLICA;DE=GERMANIA;DJ=GIBUTI;DK=DANIMARCA;DM=DOMINICA;DO=DOMINICANA, REPUBBLICA;DZ=ALGERIA;EC=ECUADOR;EE=ESTONIA;EG=EGITTO;EH=SAHARA OCCIDENTALE;ER=ERITREA;ES=SPAGNA;ET=ETIOPIA;FI=FINLANDIA;FJ=FIGI, ISOLE;FK=FALKLAND, ISOLE;FM=STATI FEDERATI DI MICRONESIA;FO=FAERÖER, ISOLE;FR=FRANCIA;GA=GABON;GB=REGNO UNITO;GD=GRENADA;GE=GEORGIA;GF=GUYANA FRANCESE;GG=GUERNSEY;GH=GHANA;GI=GIBILTERRA;GL=GROENLANDIA;GM=GAMBIA;GN=GUINEA, REPUBBLICA;GP=GUADALUPA;GQ=GUINEA EQUATORIALE;GR=GRECIA;GS=GEORGIA DEL SUD E SANDWICH DEL SUD, ISOLE;GT=GUATEMALA;GU=GUAM;GW=GUINEA-BISSAU;GY=GUIANA;HK=HONG KONG;HM=HEARD, ISOLA E MCDONALD, ISOLE;HN=HONDURAS;HR=CROAZIA;HT=HAITI;HU=UNGHERIA;ID=INDONESIA;IE=IRLANDA;IL=ISRAELE;IM=MAN, ISOLA;IN=INDIA;IO=TERRITORI BRITANNICI DELL'OCEANO INDIANO;IQ=IRAQ;IR=IRAN;IS=ISLANDA;IT=ITALIA;JE=JERSEY;JM=GIAMAICA;JO=GIORDANIA;JP=GIAPPONE;KE=KENYA;KG=KIRGHIZISTAN;KH=CAMBOGIA;KI=KIRIBATI;KM=COMORE;KN=SAINT KITTS E NEVIS;KP=COREA, REP. POP. DEM. (COREA DEL NORD);KR=COREA, REP. (COREA DEL SUD);KW=KUWAIT;KY=ISOLE CAYMAN;KZ=KAZAKISTAN;LA=LAOS;LB=LIBANO;LC=SANTA LUCIA;LI=LIECHTENSTEIN;LK=SRI LANKA;LR=LIBERIA;LS=LESOTHO;LT=LITUANIA;LU=LUSSEMBURGO;LV=LETTONIA;LY=LIBIA;MA=MAROCCO;MC=MONACO;MD=MOLDAVIA;ME=MONTENEGRO;MF=SAINT MARTIN;MG=MADAGASCAR;MH=MARSHALL, ISOLE;MK=MACEDONIA;ML=MALI;MM=BIRMANIA;MN=MONGOLIA;MO=MACAO;MP=ISOLE MARIANNE SETTENTRIONALI;MQ=MARTINICA;MR=MAURITANIA;MS=MONTSERRAT;MT=MALTA;MU=MAURITIUS;MV=MALDIVE;MW=MALAWI;MX=MESSICO;MY=MALESIA;MZ=MOZAMBICO;NA=NAMIBIA;NC=NUOVA CALEDONIA;NE=NIGER;NF=NORFOLK, ISOLA;NG=NIGERIA;NI=NICARAGUA;NL=PAESI BASSI;NO=NORVEGIA;NP=NEPAL;NR=NAURU;NU=NIUE;NZ=NUOVA ZELANDA;OM=OMAN;PA=PANAMA;PE=PERU;PF=POLINESIA FRANCESE;PG=PAPUA-NUOVA GUINEA;PH=FILIPPINE;PK=PAKISTAN;PL=POLONIA;PM=SAINT PIERRE E MIQUELON;PN=PITCAIRN;PR=PORTO RICO;PS=STATO DI PALESTINA;PT=PORTOGALLO;PW=PALAU, ISOLE;PY=PARAGUAY;QA=QATAR;RE=RÉUNION;RO=ROMANIA;RS=SERBIA;RU=RUSSIA, FEDERAZIONE;RW=RUANDA;SA=ARABIA SAUDITA;SB=SALOMONE, ISOLE;SC=SEICELLE;SD=SUDAN;SE=SVEZIA;SG=SINGAPORE;SH=TRISTAN DA CUNHA;SI=SLOVENIA;SJ=SVALBARD AND JAN MAYEN ISLANDS;SK=SLOVACCA, REPUBBLICA;SL=SIERRA LEONE;SM=SAN MARINO;SN=SENEGAL;SO=SOMALIA;SR=SURINAME;SS=MERIDIONALE SUDAN;ST=SAO TOME E PRINCIPE;SV=SALVADOR;SX=SINT MAARTEN (L'OLANDESE PART);SY=SIRIA;SZ=SWAZILAND;TC=TURKS E CAICOS;TD=CIAD;TF=TERRITORI FRANCESI DEL SUD;TG=TOGO;TH=TAILANDIA;TJ=TAGISKISTAN;TK=TOKELAU, ISOLE;TL=TIMOR EST;TM=TURKMENISTAN;TN=TUNISIA;TO=TONGA;TP=TIMOR ORIENTALE;TR=TURCHIA;TT=TRINIDAD E TOBAGO;TV=TUVALU;TW=REPUBBLICA DI CINA;TZ=TANZANIA;UA=UCRAINA;UG=UGANDA;UM=ISOLE MINORI (USA);US=STATI UNITI D'AMERICA;UY=URUGUAY;UZ=UZBEKISTAN;VA=CITTÀ DEL VATICANO;VC=SAINT VINCENT E GRENADINE;VE=VENEZUELA;VG=VERGINI, ISOLE (BRITANNICHE);VI=VERGINI, ISOLE (USA);VN=VIETNAM;VU=VANUATU;WF=WALLIS E FUTUNA;WS=SAMOA OCCIDENTALI;XK=KOSOVO;YE=YEMEN;YT=MAYOTTE;YU=SERBIA E MONTENEGRO;ZA=SUDAFRICA;ZM=ZAMBIA;ZW=ZIMBABWE"
    
    pays_list = list(x.split('=') for x in pays_list.split(';'))
    data = pd.DataFrame(pays_list, columns=['code','pays'])
    data.sort_values(by = 'pays', inplace=True)
    
    # convertir en chaine bien formée
    
    result = ""
    for item in data.iterrows() :
       r = item[1].code + "="
       r = r + item[1].pays 
       result = result + r + ";"
        
    result
