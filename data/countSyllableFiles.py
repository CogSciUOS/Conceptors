from glob import glob

syll_counts = {syllpath.split('/')[-1]: len(glob(syllpath+'/*')) for syllpath in glob('../data/birddb/syll/*')}

sorted_syll_counts = sorted(syll_counts.items(), key=lambda item: item[1], reverse=True)

with open('birddb/syllableFileCounts.txt', 'w') as f:
    for pair in sorted_syll_counts:
        f.write("{} {}\n".format(pair[0], pair[1]))

