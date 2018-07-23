#!/usr/local/bin/perl
#
# Ce script corrige la sortie du parse de Stanford pour qu'elle soit validée par l'évaluation conll
#
#


#Récupère la sortie de Stanford
($gold, $stanford) = @ARGV;


if (not defined $gold) { print("Indiquez le fichier sur lequel Stanford s'est basé en argument\n");exit();}
if (not defined $stanford) { print("Indiquez le fichier de sortie de Stanford en argument\n");exit();}


#Ouvre les fichiers
open($goldf, '<', $gold) or die "Could not open file '$gold' $!";
open($stanfordf, '<', $stanford) or die "Could not open file '$stanford' $!";

%d;$d{"ID"}=0;$d{"FORM"}=1;$d{"LEMMA"}=2;
$d{"UPOS"}=3;$d{"XPOS"}=4;$d{"FEATS"}=5;
$d{"HEAD"}=6;$d{"DEPREL"}=7;$d{"DEPS"}=8;
$d{"MISC"}=9;
$spaceAfterSentence = 0;

#Lit chaque ligne du gold
while ($g_row = <$goldf>) 
{
	#Si c'est un commentaire, out standard
  	if($g_row =~ /^#.*/)
	{
		if($spaceAfterSentence == 1){ print "\n"; }
		$spaceAfterSentence = 0;
		print $g_row;
	}
	#Si c'est une ligne conll, on la lit
	elsif($g_row =~ /(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)/)
	{
		$spaceAfterSentence = 1; #Mettre un espace avant la prochaine phrase
		@g_conll = ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10); #Récupère les données du regex
		
		#Si c'est un mot décomposable (du => de le), out standard
		if($g_conll[$d{"ID"}] =~ /.*-.*/)
		{
			print join("\t",@g_conll)."\n";
			next;
		}#Sinon si ce n'est pas standard (pas traité par le système)
		elsif($g_conll[$d{"ID"}] =~ /[0-9]+\.[0-9]+/)
		{
			print join("\t",@g_conll)."\n";
			next;
		}#Sinon on va chercher à s'aligner avec le systeme
		else{
			$gardefou = 0;
			#print("enter loop\n");
			do
			{
				#print("in loop\n");
				#regex pour le systeme
				do
				{
					$s_row = <$stanfordf>;
					@s_conll = split /\t/,$s_row;
				}
				while(scalar @s_conll != 10);
				#print("$s_row");
				#print("$g_row");
				#print($g_conll[$d{"ID"}] . " " . $s_conll[$d{"ID"}] . "? \n");
				if("$s_conll[$d{ID}]" eq "$g_conll[$d{ID}]") #Si les ID correspondent, on est aligné
				{
					#print("ok\n");
					$g_conll[$d{"UPOS"}] = $s_conll[$d{"UPOS"}];
					#$g_conll[$d{"XPOS"}] = $s_conll[$d{"XPOS"}];
					$g_conll[$d{"FEATS"}] = $s_conll[$d{"FEATS"}];
					$g_conll[$d{"HEAD"}] = $s_conll[$d{"HEAD"}];
					$g_conll[$d{"DEPREL"}] = $s_conll[$d{"DEPREL"}];
					print join("\t",@g_conll)."\n";
					next;
				}
		
				if($gardefou > 100)
				{
					print("ERROR STANFORD_FIX LOOP\n");
					exit(1);
				}
				$gardefou = $gardefou+1;
			}while( $s_row =~ /.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*\t.*/); #Sinon on cherche la correspondance sur les prochaines lignes
		}
	}
}
print "\n";












