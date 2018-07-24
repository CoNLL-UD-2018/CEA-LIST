#!/usr/local/bin/perl
#
# Ce script permet de désambiguiser les tags issus du wiktionary grâce à un alignement avec une autre langue à annotations connues
# Il permet également de construire un dictionnaire bilingue
#
use open ':std', ':encoding(UTF-8)';
binmode STDOUT, ":utf8";
use open ":encoding(utf8)";
use feature 'unicode_strings';
use utf8;


($align, $file1, $file2, $mode, $prior_knowledge_file) = @ARGV;

$maxsent = 1000000;

$write_file2 = $mode =~ /write_desambiguate/ ? 1 : 0;
$write_file2x = $mode eq "write_desambiguate_xtag" ? 1 : 0;
$debug_mode = $mode eq "debug" ? 1 : 0;
$stats_upos = $mode eq "stats" ? 1 : 0;
$dic_w1_w2 = $mode eq "dic" ? 1 : 0;

open($alignf, '<', $align) or die "Could not open file '$align' $!";
open($file1f, '<', $file1) or die "Could not open file '$file1' $!";
open($file2f, '<', $file2) or die "Could not open file '$file2' $!";

#Si on a un fichier qui contient une connaissance à priori sur les vraies proportions (issue d'un gold ud)
%prior_upos_prop;
if(-e $prior_knowledge_file)
{
	@upos_name = `cat $prior_knowledge_file|cut -f1`;
	@upos_prop = `cat $prior_knowledge_file|cut -f2`;
	for($i =0;$i<scalar @upos_name; $i = $i+1)
	{
		$upos_name[$i] =~ s/\R//g;
		$upos_name[$i] =~ s/\s+$//;
		$upos_prop[$i] =~ s/\R//g;
		$upos_prop[$i] =~ s/\s+$//;
		$prior_upos_prop{$upos_name[$i]} = $upos_prop[$i];
		#print("$upos_name[$i] $upos_prop[$i] $prior_upos_prop{$upos_name[$i]}\n");
	}
}

%d;$d{"ID"}=0;$d{"FORM"}=1;$d{"LEMMA"}=2;
$d{"UPOS"}=3;$d{"XPOS"}=4;$d{"FEATS"}=5;
$d{"HEAD"}=6;$d{"DEPREL"}=7;$d{"DEPS"}=8;
$d{"MISC"}=9;
$nsent = 0;
$word_count = 0;
$upos_ok = 0;
$upos_in_xpos = 0;
$xpos_occasion = 0;
$sent_id = 0;
%dic_upos_error = ();
%dic_occ = ();
%dic_w1_w2 = ();
while($align = <$alignf>)
{
	#print($align);
	@words1 = ();
	@upos1 = ();
	@xpos1 = ();
	while($rowfile1 = <$file1f>)
	{
		$rowfile1 =~ s/\R//g;
		#print("$rowfile1\n");
		if($rowfile1 =~ /(.*)\t(.*)\t(.*)\t(.*)/)
		{
			@row = split("\t",$rowfile1);
			push(@words1,$row[$d{"FORM"}]);
			push(@upos1,$row[$d{"UPOS"}]);
			push(@xpos1,$row[$d{"XPOS"}]);
		}
		if($rowfile1 !~ /# sent_id = 1$/ and $rowfile1 =~ /# sent_id/)
		{
			last;
		}
	}

	@words2 = ();
	@upos2 = ();
	@xpos2 = ();
	@spaceafter2 = ();
	while($rowfile2 = <$file2f>)
	{
		$rowfile2 =~ s/\R//g;
		#print("$rowfile2\n");
		if($rowfile2 =~ /(.*)\t(.*)\t(.*)\t(.*)/)
		{
			@row = split("\t",$rowfile2);
			push(@words2,$row[$d{"FORM"}]);
			push(@upos2,$row[$d{"UPOS"}]);
			push(@xpos2,$row[$d{"XPOS"}]);
			push(@spaceafter2,$row[$d{"MISC"}]);
		}
		if(($rowfile2 !~ /# sent_id: 1$/ and $rowfile2 !~ /# sent_id = 1$/) and $rowfile2 =~ /# sent_id/)
		{
			last;
		}
	}


	$score_align = 0;
	if($align =~ /(.*)\|\|\|(.*)/)
	{
		$align = $1;
		$score_align = $2;
		$score_align /= scalar @words1;
	}
	
	#print(join(' ',@words1)."\n");
	#print(join(' ',@words2)."\n");
	$l1 = scalar @words1;
	$l2 = scalar @words2;
	if($l2 == 0 or $l1 == 0)
		{last;}

	#seuil de qualité (score d'alignement et différence de taille phrases,
	#if($score_align > -5 and abs($l1-$l2) <= 4)
	@aligns = split(' ',$align);
	$n_align = scalar @aligns;
	if( ($l1 + $l2) > 0)
		{$prop_align = ($n_align * 2) / ($l1 + $l2);}
	else
		{$prop_align = 0;}
	# ou, pourcentage de mot alignés)
	#print("$n_align, $l1, $l2\n");
	if($prop_align > 0.7 or $dic_w1_w2 or $stats_upos or $debug_mode)
	{
		#mode debug
		if($debug_mode)
		{
			print(join('__',@words1)."\n");
			print(join('__',@words2)."\n");
		}

		
		#Si on a une prior knowledge sur les stats
		if(-e $prior_knowledge_file)
		{	
			#pour chaque mot
			for($i = 0; $i < $l2; $i+=1)
			{
				$w2 = $words2[$i];
				$u2 = $upos2[$i];
				$x2 = $xpos2[$i];
				@xtags = split /\|/,$x2;
				$ntag = scalar @xtags;
				
				#Si plusieurs choix sont possibles
				if($ntag > 1)
				{
					#cumule les probas
					$p=0;
					@probs = ($p);
					for $tag(@xtags)
					{
						#print("prior:$tag : $prior_upos_prop{$tag}\n");
						$p += $prior_upos_prop{$tag};
						push(@probs,$p);
					}

					#choisit
					$r = rand($p);
					#print("$w2 - 0 -> $p : $r\n");
					for($j=0;$j<$ntag; $j+=1)
					{
						$p1= $probs[$j];
						$p2 = $probs[$j+1];
						#print("$r - $p1 - $p2 - $xtags[$j]\n");
						if($r >= $p1 and $r < $p2)
						{
							$upos2[$i] = $xtags[$j];
							last;
						}
					}
					#print("\n");
				}
			}
		}


		#pour chaque alignement mot1->mot2
		foreach $al (@aligns)
		{
			@a = split('-',$al);
			$a1 = @a[0];
			$a2 = @a[1];

			if($debug_mode)
			{
				print("$a1 $a2\n");
				print("$words1[$a1] ($upos1[$a1]) - $words2[$a2] ($upos2[$a2] | $xpos2[$a2])\n");
			}
			
			#compte le nombre d'occurence des mots pour normaliser les stats
			$dic_occ{$words2[$a2]} += 1;
			

			#si même tag
			if($upos1[$a1] eq $upos2[$a2])
			{
				$upos_ok += 1;
				if(($words1[$a1] !~ /[\p{P}]+/ and $words2[$a2] !~ /[\p{P}]+/) or ($words1[$a1] =~ /[\p{P}]+/ and $words2[$a2] =~ /[\p{P}]+/))
					{$dic_w1_w2{"$words2[$a2]\t$words1[$a1]"} += 1;}

				#projette le xtag ko->th
				if($write_file2x)
					{$newxpos[$a2] = $xpos1[$a1];}
			}
			else
			{	
				
				#si tag différent mais le wiktionary contenait un autre tag qui colle (et tag non vide) OU si on a un tag sans information (upos=X)
				if($xpos2[$a2] =~ /.*$upos1[$a1].*/ and length($upos1[$a1]) > 1 or $upos2[$a2] eq "X")
				{
					$upos2[$a2] = $upos1[$a1];
					$upos_in_xpos += 1;
					$upos_ok += 1;
					if(($words1[$a1] !~ /[\p{P}]+/ and $words2[$a2] !~ /[\p{P}]+/) or ($words1[$a1] =~ /[\p{P}]+/ and $words2[$a2] =~ /[\p{P}]+/))
						{$dic_w1_w2{"$words2[$a2]\t$words1[$a1]"} += 1;}
					
					#projette le xtag ko->th
					if($write_file2x)
						{$newxpos[$a2] = $xpos1[$a1];}
				}
				#si aucun tag ne correspond
				else
				{
					#si on calcule les stats sur les upos qui divergent avec l'alignement
					if($stats_upos)
					{
						#filtre qualité sur les mauvais alignements de symboles
						if($upos1[$a1] ne "PUNCT" and $upos1[$a1] ne "SYM" and $upos1[$a1] ne "NUM" and $upos1[$a1] ne "CCONJ" and
							$upos2[$a2] ne "PUNCT" and $upos2[$a2] ne "SYM" and $upos2[$a2] ne "NUM" and $upos2[$a2] ne "CCONJ")
						{
							$dic_upos_error{"$words2[$a2] $upos1[$a1]"} += 1;
						}
						#print("$words2[$a2] $upos1[$a1]\n");
					}

					#print("$upos1[$a1] $upos2[$a2]\n");
				}
				
				$xpos_occasion += 1;
			}
			$word_count += 1;
		}
		$prop1 = $upos_ok/$word_count;
		if($xpos_occasion != 0)
			{$prop2 = $upos_in_xpos/$xpos_occasion;}
		else
			{$prop2 = 1;}
		$prop1 = sprintf("%.2f",$prop1*100);
		$prop2 = sprintf("%.2f",$prop2*100);
		if($debug_mode)
		{
			print("upos_ok = $prop1%   xpos_ok = $prop2% $word_count $score_align\n");
			print("\n");
			print("\n");
		}
		#calcule le nombre de non-corrigé
		$n_xtag_uncorrected = 0;
		if($write_file2x)
		{
			for($j=0;$j < $l2; $j+=1)
			{
				$x2 = $xpos2[$j];
				$w2 = $words[$j];
				if($x2 eq "_" or $x2 =~ /\|/ or ( $x2 =~ /[\p{P}]+/ and $w2 !~ /[\p{P}]+/ ) )
					{$n_xtag_uncorrected += 1;}
			}
		}

		#ecrit le thai désambiguisé
		$prop_xtaguncor = ($n_xtag_uncorrected / $l2);
		if($write_file2 and (!$write_file2x or $prop_xtaguncor < 0.4))
		{
			$sent_id += 1;
			print("# sent_id = $sent_id  [$n_xtag_uncorrected,$l2,$prop_xtaguncor]\n");
			print("# tokenized_sent = ".join('  ',@words2)."\n");
			for($i=0;$i<scalar @words2;$i+=1)
			{
				$id = $i+1;

				#no xtags				
				#$x = "_";

				#xtag
				$x = $xpos2[$i];

				if($newxpos[$i])
					{$x = $newxpos[$i];}
				print("$id\t$words2[$i]\t$words2[$i]\t$upos2[$i]\t$x\t_\t_\t_\t_\t$spaceafter2[$i]\n");
			}
			print("\n");
		}

	}
	$nsent += 1;
	if($nsent > $maxsent)	
		{last;}
}


#affiche les stats sur le pourcentage de fois où un mot n'a pas un tag cohérent avec l'alignement (filtrage exclusif efmaral nullprior)
$min_n_occurence = 10;
$min_percent_modified = 0.15;
if($stats_upos)
{
	foreach $wordupos (keys %dic_upos_error)
	{
		#pourcentage de fois où le mot a un tag différent entre langue1 et langue2
		@worduposarray = split / /,$wordupos;
		#print("$wordupos - $worduposarray[0] - $worduposarray[1] - $dic_occ{$worduposarray[0]} -\n");
		$s = $dic_upos_error{$wordupos}/$dic_occ{$worduposarray[0]};

		#filtre qualité
		if($dic_occ{$worduposarray[0]} > $min_n_occurence and $s > $min_percent_modified)
		{
			$s = sprintf("%.0f",$s*100);
			print("$wordupos $s\n");
		}
	}
}

#Affiche pour chaque couple w1 w2 proposés dans l'alignement le nombre d'occurence de l'alignement w1 w2
$min_n_occurence = 1;
if($dic_w1_w2)
{
	foreach $w12 (keys %dic_w1_w2)
	{
		if($dic_w1_w2{$w12} > $min_n_occurence)
			{print("$w12\n");}
	}
}


