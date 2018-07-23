#!/usr/local/bin/perl
#
#	Check if a conllu file contains projective arcs
#



#Take the name of the conllu file as argument
($file) = @ARGV;


if (not defined $file) { print("Please indicate the filename.\n");exit();}


#Open the file
open($filef, '<', $file) or die "Could not open file '$gold' $!";

%d;$d{"ID"}=0;$d{"FORM"}=1;$d{"LEMMA"}=2;
$d{"UPOS"}=3;$d{"XPOS"}=4;$d{"FEATS"}=5;
$d{"HEAD"}=6;$d{"DEPREL"}=7;$d{"DEPS"}=8;
$d{"MISC"}=9;

#Read each line
@ids;
@parents;
while ($line = <$filef>) 
{
	#If the line is in the conllu format
	if($line =~  /(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)\t(.*)/)
	{
		@conll = ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10);
		$id = @conll[$d{"ID"}];
		if($id =~ /^[0-9]+$/)
		{
			$parent = @conll[$d{"HEAD"}];
			$str = $parent;

			#save the id and the pid
			push(@ids, $id);
			push(@parents, $parent);
		}
	}
	#When we reached the end of the sentence
	else
	{
		#We can't have two crossing intervals [id,pid]
		for($i =0; $i < scalar @ids; $i++)
		{
			for($j =0; $j < scalar @ids; $j++)
			{
				#For two different intervals
				if($i != $j)	
				{
					$id1 = $ids[$i];
					$id2 = $ids[$j];
					
					$min_id = ($id1, $id2)[$id1 > $id2];	
					$left = $j;
					$right = $i;
					if($id1 == $min_id) { $left = $i; $right = $j;}

					#id1 contains the leftmost word (could be by default as we're pushing ids in the default order)
					$id1 = $ids[$left];
					$parent1 = $parents[$left];
					$id2 = $ids[$right];
					$parent2 = $parents[$right];
					
					#example: [1,3] [2,4] : 2 < 3 && 4 > 3 -> crossing -> nonprojective
					if($id2 < $parent1 && $parent2 > $parent1)
					{
						#print("$id1,$parent1 crossing $id2,$parent2 \n");
						print("$file nonprojective\n");
						exit(0);
					}
				}
			}
		}

		@ids=(); 
		@parents=(); 
		#print("newsent\n");
	}
			
}
print "$file projective\n";












