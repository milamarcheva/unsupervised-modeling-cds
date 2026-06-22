#!/usr/bin/perl
#
use strict;
if (@ARGV != 2) { die "usage $0 <system-output-file> <gold-file>\n"; }

my %A; 
my %S; 
my %G; 
my %M;
my $line;
my $nline = 0;
my @slines;
my @glines;
my $total = 0;
my $correct = 0;

open S, "<$ARGV[0]" or die "can not open $ARGV[0].\n";
open G, "<$ARGV[1]" or die "can not open $ARGV[1].\n";
while ($line = <S>) {
  $nline++;
  chomp $line;
  push @slines, $line;
  my @stags = split /\s+/, $line;

  $line = <G>;
  chomp $line;
  push @glines, $line;
  my @gtags = split /\s+/, $line;

  die "numbers of tokens don't match at line $nline" if ($#stags != $#gtags);
  for (my $i = 0; $i <= $#gtags; $i++) {
    $A{$stags[$i]}{$gtags[$i]} ++;
    $S{$stags[$i]} ++;
    $G{$gtags[$i]} ++;
  }
  foreach my $tag (keys %S) {
    my $best = 0;
    my $best_tag = "";
    foreach my $match (keys %{$A{$tag}}) {
      if ($A{$tag}{$match} > $best) {
        $best = $A{$tag}{$match};
        $best_tag = $match;
      }
    }
    $M{$tag} = $best_tag;
  }
}
close S;
close G;

for (my $i = 0; $i <= $#slines; $i++) {
  my @stags = split /\s+/, $slines[$i];
  my @gtags = split /\s+/, $glines[$i];

  for (my $j = 0; $j <= $#stags; $j++) {
    $total ++;
    if ($M{$stags[$j]} eq $gtags[$j]) {
      $correct ++;
    }
  }
}

printf "ACC: %3.2f (%d/%d)", $correct/$total, $correct, $total,
