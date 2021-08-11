import torch
from sklearn.datasets import fetch_20newsgroups
import textwrap
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import BertTokenizer


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Retrieve the dataset.
# The scikit-learn example recommends stripping the metadata from these examples
# because it can be too much of a give-away to make it an interesting text
# classification problem. That is what the `remove` parameter is doing below.
train = fetch_20newsgroups(subset='train',
                           remove=('headers', 'footers', 'quotes'))

test = fetch_20newsgroups(subset='test',
                          remove=('headers', 'footers', 'quotes'))

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)

# Randomly choose some examples.
for i in range(10):
    # Choose a random sample by index.
    j = random.choice(range(len(train.data)))

    # Get the text as 'x' and the label integer as 'y'.
    x = train.data[j]
    y = train.target[j]

    # Print out the name of the category and the text.
    print('')
    print('========', train.target_names[y], '========')
    print(wrapper.fill(x))
    print('')

sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (10, 5)

# Plot the number of tokens of each length.
sns.countplot(train.target)
plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('# of Training Samples')
# plt.show()

print(train.target_names[19])

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# Record the length of each sequence (in terms of BERT tokens).
lengths = []

print('Tokenizing comments...')

# For every sentence...
for sen in train.data:

    # Report progress.
    if (len(input_ids) % 20000) == 0:
        print('  Read {:,} comments.'.format(len(input_ids)))

    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sen,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        # max_length = 512,          # Truncate all sentences.
        # return_tensors = 'pt',     # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

    # Record the non-truncated length.
    lengths.append(len(encoded_sent))

print('DONE.')
print('{:>10,} comments'.format(len(input_ids)))

print('   Min length: {:,} tokens'.format(min(lengths)))
print('   Max length: {:,} tokens'.format(max(lengths)))
print('Median length: {:,} tokens'.format(np.median(lengths)))

num_over = 0

# For all of the length values...
for length in lengths:
    # Tally if it's over 512.
    if length > 512:
        num_over += 1

print('{:,} of {:,} comments will be truncated ({:.2%})'
      .format(num_over, len(lengths), float(num_over) / float(len(lengths))))

