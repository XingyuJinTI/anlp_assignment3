from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  return log((N*c_xy)/(c_x*c_y),2);

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.
  dot_product = 0
  v0_size = 0
  v1_size = 0
    
  for co_occur_id, ppmi in v0.items():
      v0_size += v0[co_occur_id]**2
      if co_occur_id in v1:
        dot_product += v0[co_occur_id] * v1[co_occur_id]
        
  for co_occur_id, ppmi in v1.items():
      v1_size += v1[co_occur_id]**2
      
  v0_size = v0_size**0.5
  v1_size = v1_size**0.5
  
  return dot_product/(v0_size*v1_size)

def euclidean_distance(v0,v1,wid0,wid1):

  ids_in_both = {}
  ids_in_both = set()
  
  for co_occur_id, ppmi in v0.items():
      ids_in_both.add(co_occur_id)
        
  for co_occur_id, ppmi in v1.items():
      ids_in_both.add(co_occur_id)    
    
  if wid0 in ids_in_both:
      ids_in_both.remove(wid0)
      
  if wid1 in ids_in_both:
      ids_in_both.remove(wid1)
      
  square_distance = 0
    
  for wid in ids_in_both:
      if wid not in v0:
          v0_elem = 0
      else:
          v0_elem = v0[wid]
          
      if wid not in v1:
          v1_elem = 0
      else:
          v1_elem = v1[wid]
        
      square_distance += (v1_elem - v0_elem)**2
        
  square_distance = square_distance**0.5
  
  return 1/square_distance

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    
    vectors = {}
    for wid0 in wids:
        vectors[wid0] = {}
        
        for co in co_counts[wid0]:
            vectors[wid0][co] = max(0, PMI(co_counts[wid0][co], o_counts[wid0], o_counts[co], tot_count))

    return vectors

def create_ppmi_laplace_vectors(wids, o_counts, co_counts, tot_count, laplace_constant):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    
    for wid0 in wids:
        for o_count in o_counts:
            if o_count not in co_counts[wid0]:
                co_counts[wid0][o_count] = laplace_constant
            else:
                co_counts[wid0][o_count] += laplace_constant
    
    tot_count_plus_2v = tot_count + laplace_constant*len(o_counts)
        
    vectors = {}
    for wid0 in wids:
        vectors[wid0] = {}
        
        for co in co_counts[wid0]:
            vectors[wid0][co] = max(0, PMI(co_counts[wid0][co], o_counts[wid0], o_counts[co], (tot_count**2)/tot_count_plus_2v))

    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.8f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))

def freq_v_sim(sims):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
    #xs.append(c0+c1)
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]

test_word_pairs = [["frog", "bean"], ["test", "word"], ["bieber", "@justinbieber"], ["mum", "mom"], ["favour", "favor"], ["sofa", "couch"], ["advert", "commercial"], ["petrol", "gas"], ["honour", "honor"], ["lecturer", "professor"], ["revise", "study"]]
#test_words = ["mum", "mom", "bieber"]
test_words = ["gate",	"expansion",	"shark",	"small",	"copper",	"insist",	"cabinet",	"donor",	"deer",	"pest",	"firefighter",	"woman",	"suit",	"absorption",	"entry",	"tribute",	"neutral",	"angle",	"common",	"utter",	"replace",	"resignation",	"aunt",	"forget",	"fade",	"mix",	"important",	"frown",	"house",	"vacuum",	"multimedia",	"dictate",	"bolt",	"bush",	"bundle",	"incongruous",	"pattern",	"diplomat",	"score",	"function",	"slam",	"agile",	"pull",	"looting",	"peel",	"hilarious",	"bed",	"wind",	"wreck",	"fuel",	"fastidious",	"reality",	"agent",	"era",	"extension",	"keep",	"raise",	"stereotype",	"youth",	"faith",	"urge",	"post",	"grudge",	"flag",	"underline",	"demonstrate",	"feminist",	"cattle",	"battery",	"trail",	"manager",	"engineer",	"design",	"door",	"insight",	"position",	"high",	"train",	"medium",	"work",	"arch",	"sail",	"love",	"heaven",	"origin",	"growth",	"play",	"popular",	"quaint",	"stab",	"dish",	"radiation",	"follow",	"beef",	"snail",	"menu",	"dome",	"identification",	"gaffe",	"sour"]
stemmed_words = [tw_stemmer(w) for w in test_words]
stemmed_word_pairs = [[tw_stemmer(w) for w in test_word_pair] for test_word_pair in test_word_pairs]

all_wids = set()
for stemmed_pair in stemmed_word_pairs:
    for stemmed_word in stemmed_pair:
        all_wids.add(word2wid[stemmed_word])

#all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = [(word2wid[tw_stemmer(x)], word2wid[tw_stemmer(y)]) for x, y in test_word_pairs]
#wid_pairs = make_pairs(all_wids)

#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
#vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
vectors = create_ppmi_laplace_vectors(all_wids, o_counts, co_counts, N, 0.1)

# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
#c_sims = {(wid0,wid1): euclidean_distance(vectors[wid0],vectors[wid1],wid0,wid1) for (wid0,wid1) in wid_pairs}

freq_v_sim(c_sims)

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)
