
use open ':std', ':encoding(UTF-8)';
binmode STDOUT, ":utf8";
use open ":encoding(utf8)";
use feature 'unicode_strings';
use utf8;




($f1, $f2) = @ARGV;

open(my $fh1, '<:encoding(UTF-8)', $f1)
  or die "Could not open file '$f1' $!";

open(my $fh2, '<:encoding(UTF-8)', $f2)
  or die "Could not open file '$f2' $!";


while($r1 = <$fh1>)
{
	$r2 = <$fh2>;
	$r1 =~ s/\R//g;
	$r2 =~ s/\R//g;
	print("$r1 ||| $r2\n");
}



