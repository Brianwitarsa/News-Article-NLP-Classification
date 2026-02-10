# News Article Routing with Classical ML + CNN (AG News)
Summary

Starting with 120,000 training samples and 7,600 test samples, we approached this as a sports news platform that just acquired a large bundle of mixed news articles and needs to separate Sports stories from everything else (Business + World + Sci/Tech). Early EDA showed this split was promising as vocabulary analysis shows strong results. Only 4 of the top 50 words overlapped, suggesting Sports articles use noticeably different language than non-Sports articles.

For modeling, we compared a classical baseline (TF–IDF with unigrams/bigrams → Truncated SVD (300 dims) → Random Forest) against a deep learning approach (1D CNN with embeddings and global max pooling). The Random Forest achieved strong performance, but the CNN improved results further, reaching 98.37% test accuracy and ~99.1% precision for the non-Sports class (and ~96.1% precision for Sports). These results show that sequence-based CNN models can capture phrase-level patterns more effectively than compressed TF–IDF features, leading to more reliable article routing decisions.

Links: 
- [Code]([https://github.com/Brianwitarsa/New-Article_Classification/blob/main/Brian_Witarsa_AML_Final.ipynb])
