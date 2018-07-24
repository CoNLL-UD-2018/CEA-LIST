#!/usr/local/bin/perl
#




do "load_config.pl";


($conllu, $nsent) = @ARGV;



#Ouvre le fichier
open($f, '<', $conllu) or die "Could not open file '$train_data' $!";


#Sépare les fichiers
$n = 0;
while ($row = <$f>) 
{
	#Compte
	if($row =~ /^1\t/){$n++;}


	if($n >= $nsent && $row eq "\n"){print("\n");exit(0);}	

	print $row; 
	
}











