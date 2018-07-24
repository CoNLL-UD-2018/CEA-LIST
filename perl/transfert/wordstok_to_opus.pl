
use open ':std', ':encoding(UTF-8)';
binmode STDOUT, ":utf8";
use open ":encoding(utf8)";
use feature 'unicode_strings';
use utf8;


$frow = 0;
while($row = <>)
{
	$row =~ s/\R//g;
	#print("$row\n");
	if($row =~ /^# sent/ && $frow == 1)
	{
		print("\n");
	}
	else
	{
		if($row !~ /^#/)
		{
			print("$row ");
			$frow = 1;
		}
	}
}	
print("\n");





