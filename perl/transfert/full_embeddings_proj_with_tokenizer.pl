#!/usr/local/bin/perl
#
#  Ce script a pour but d'effectuer toute la chaine de transfert d'une langue cible dotée vers une langue source peu dotée en créant un corpus annoté pour cette langue source 
#
#

# polish -> czech
# EXAMPLE: perl full_embeddings_proj_with_tokenizer.pl ../pl-cz/model ../pl-cz/pl.short ../pl-cz/cs.short ../../../CoNLL2018/models/udpipe-models/polish-lfg-ud-2.2-conll18-180430.udpipe ../../../CoNLL2018/models/udpipe-models/czech-cac-ud-2.2-conll18-180430.udpipe ../../../efmaral/align.py ../embeddings_proj/main.py ../../Embeddings18/Polish/pl.vectors ../../Embeddings18/Czech/cs.vectors ../../Train2018/ud-treebanks-v2.2/UD_Polish-LFG/ ../../Train2018/ud-treebanks-v2.2/UD_Czech-CAC/

($savedir, $source_opus, $target_opus, $udpipemodel1, $udpipemodel2, $efmaral, $embprojpy, $embsrc, $embtgt, $ud1, $ud2) = @ARGV;
print("\n");

#création du savedir
if(not -e $savedir)
{
	$cmd = "mkdir $savedir";
	print("$cmd\n");
	system($cmd);
}

#Tokenization langue 1
$l1end = "l1.conllu";
$l1 = "$savedir/$l1end";
if(not -e $l1)
{
	$cmd = "udpipe --tokenize --tokenizer=presegmented $udpipemodel1 $source_opus > $l1";
	print("$cmd\n");
	system($cmd);
}
#cut1
$cut1 = "$savedir/cut1.txt";
if(not -e $cut1)
{
	$cmd = "cat $l1| cut -f2 > $cut1";
	print("$cmd\n");
	system($cmd);
}

#l1full.txt
$l1full = "$savedir/l1full.txt";
if(not -e $l1full)
{
	$cmd = "cat $cut1 |perl wordstok_to_opus.pl > $l1full";
	print("$cmd\n");
	system($cmd);
}




#Tokenization langue 2
$l2end = "l2.conllu";
$l2 = "$savedir/$l2end";
if(not -e $l2)
{
	$cmd = "udpipe --tokenize --tokenizer=presegmented $udpipemodel2 $target_opus > $l2";
	print("$cmd\n");
	system($cmd);
}

#cut2
$cut2 = "$savedir/cut2.txt";
if(not -e $cut2)
{
	$cmd = "cat $l2| cut -f2 > $cut2";
	print("$cmd\n");
	system($cmd);
}

#l2full.txt
$l2full = "$savedir/l2full.txt";
if(not -e $l2full)
{
	$cmd = "cat $cut2 |perl wordstok_to_opus.pl > $l2full";
	print("$cmd\n");
	system($cmd);
}





#paste
$opusfull = "$savedir/opus_full.txt";
if(not -e $opusfull)
{
	$cmd = "perl paste.pl $l1full $l2full >$opusfull";
	print("$cmd\n");
	system($cmd);
}


#efmaral
$forward80 = "$savedir/fwd80.align";
if(not -e $forward80)
{
	$cmd = "$efmaral -i $opusfull -v --null-prior 0.8>$forward80 ";
	print("$cmd\n");
	system($cmd);
}

#efmaral
$forward50 = "$savedir/fwd50.align";
if(not -e $forward50)
{
	$cmd = "$efmaral -i $opusfull -v --null-prior 0.5>$forward50 ";
	print("$cmd\n");
	system($cmd);
}
#efmaral
$forward95 = "$savedir/fwd95.align";
if(not -e $forward95)
{
	$cmd = "$efmaral -i $opusfull -v --null-prior 0.95>$forward95 ";
	print("$cmd\n");
	system($cmd);
}


#genere le dictionnaire 1->2
$dico12 = "$savedir/dico12.txt";
if(not -e $dico12)
{
	$cmd = "perl desambiguate_upos.pl $forward95 $l2 $l1 dic > $dico12";
	print("$cmd\n");
	system($cmd);
}


#transfère les embeddings dans le repère 2
$nembed = 100000;
$embed12 = "$savedir/embed12.vec";
if(not -e $embed12)
{
	$cmd = "python $embprojpy $embsrc $embtgt $dico12 $nembed > $embed12";
	print("$cmd\n");
	system($cmd);
}

#Mode de génération de données de train et dev par fusion pour apprentissage
if($ud1 and $ud2)
{
	#fusionne les train des deux corpus
	$minsent = 1500;
	$traindata = "$savedir/train.conllu";
	if(not -e $traindata)
	{
		$train1 = "$ud1/".`ls $ud1 |grep "train.conllu"`; $train1=~ s/\R//g;
		$train2 = "$ud2/".`ls $ud2 |grep "train.conllu"`; $train2=~ s/\R//g;
		$n1 = `cat $train1 |grep root |wc -l`; $n1=~ s/\R//g;
		$n2 = `cat $train2 |grep root |wc -l`; $n2=~ s/\R//g;
		$nsent = ($n1, $n2)[$n1 > $n2];
		if($nsent < $minsent)
			{$nsent = $minsent;}
		print("$nsent phrases de train/treebank\n");
		`perl cut_n_sentences_from_conllu.pl $train1 $nsent > $savedir/tmp1`;
		`perl cut_n_sentences_from_conllu.pl $train2 $nsent > $savedir/tmp2`;
		$cmd = "cat $savedir/tmp1 $savedir/tmp2 > $traindata";
		print("$cmd\n");
		system($cmd);
		`rm $savedir/tmp1`;
		`rm $savedir/tmp2`;
	}


	#fusionne les dev des deux corpus
	$minsent = 500;
	$devdata = "$savedir/dev.conllu";
	if(not -e $devdata)
	{
		$dev1 = "$ud1/".`ls $ud1 |grep "dev.conllu"`; $dev1=~ s/\R//g;
		$dev2 = "$ud2/".`ls $ud2 |grep "dev.conllu"`; $dev2=~ s/\R//g;
		$n1 = `cat $dev1 |grep root |wc -l`; $n1=~ s/\R//g;
		$n2 = `cat $dev2 |grep root |wc -l`; $n2=~ s/\R//g;
		$nsent = ($n1, $n2)[$n1 > $n2];
		if($nsent < $minsent)
			{$nsent = $minsent;}
		print("$nsent phrases de dev/treebank\n");
		`perl cut_n_sentences_from_conllu.pl $dev1 $nsent > $savedir/tmp1`;
		`perl cut_n_sentences_from_conllu.pl $dev2 $nsent > $savedir/tmp2`;
		$cmd = "cat $savedir/tmp1 $savedir/tmp2 > $devdata";
		print("$cmd\n");
		system($cmd);
		`rm $savedir/tmp1`;
		`rm $savedir/tmp2`;
	}

	#merge embeddings
	$embedmerge = "$savedir/merged.vec";
	if(not -e $embedmerge)
	{
		$nembedm1 = $nembed-1;
		`head -n $nembed $embtgt |tail -n $nembedm1 > $savedir/tgt_embed$nembed.vec`;
		$cmd = "cat $embed12 $savedir/tgt_embed$nembed.vec > $embedmerge";
		print("$cmd\n");
		system($cmd);
	}


}










