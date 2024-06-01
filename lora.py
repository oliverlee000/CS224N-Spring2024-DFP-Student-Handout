import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *
import math

LORA_SIZE = 10

# To test, just use 10 examples, and see whether the training loss goes to 0. If it does, then the model is training, and everything is going ok

# Make sure to keep the pretrained weights; otherwise, LoraDora should decrease parameters

'''
Returns function that is equivalent to applying Lora layers W_0, A, and B

W_0: initial pretrained weights, Linear layer
A: first parameter from Lora, Linear layer
B: second parameter from Lora, Linear layer
'''
def lora_dense(W0, A, B):
  def apply_lora(x):
    return W0(x) + B(A(x))
  return apply_lora

'''
Lora-fied version of BertSelfAttention
'''
class LoraBertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.query_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.query_B = nn.Linear(LORA_SIZE, self.all_head_size)
    self.query.requires_grad = False

    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.key_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.key_B = nn.Linear(LORA_SIZE, self.all_head_size)
    self.key.requires_grad = False

    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    self.value_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.value_B = nn.Linear(LORA_SIZE, self.all_head_size)
    self.value.requires_grad = False

    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    # TODO
    B, nh, T, hs = query.size()
    attn_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1))) # size (B, nh, T, T)
    # Adding attention_mask
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = attn_weights @ value # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, nh * hs) # re-assemble all head outputs side by side
    return attn_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    # Dora transform key, value, query
    key = lambda x: self.key(x) + self.key_B(self.key_A(x))
    value = lambda x: self.value(x) + self.value_B(self.value_A(x))
    query = lambda x: self.query(x) + self.query_B(self.query_A(x))

    key_layer = self.transform(hidden_states, key)
    value_layer = self.transform(hidden_states, value)
    query_layer = self.transform(hidden_states, query)
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value

'''
Lora-fied version of BertLayer
'''
class LoraBertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = LoraBertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_dense_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.attention_dense_B = nn.Linear(LORA_SIZE, config.hidden_size)
    self.attention_dense.requires_grad = False

    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_dense_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.interm_dense_B = nn.Linear(LORA_SIZE, config.intermediate_size)
    self.interm_dense.requires_grad = False

    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_dense_A = nn.Linear(config.intermediate_size, LORA_SIZE)
    self.out_dense_B = nn.Linear(LORA_SIZE, config.hidden_size)
    self.out_dense.requires_grad = False

    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.
    # TODO
    output = dense_layer(output)
    output = dropout(output)
    output = ln_layer(output + input)
    return output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # Lora transform
    attention_dense = lambda x: self.attention_dense(x) + self.attention_dense_B(self.attention_dense_A(x))
    interm_dense = lambda x: self.interm_dense(x) + self.interm_dense_B(self.interm_dense_A(x))
    out_dense = lambda x: self.out_dense(x) + self.out_dense_B(self.out_dense_A(x))


    attn_output = self.self_attention(hidden_states, attention_mask)
    add_norm_output = self.add_norm(hidden_states, attn_output, attention_dense, self.attention_dropout, self.attention_layer_norm)
    feed_forward_output = self.interm_af(interm_dense(add_norm_output))
    add_norm_output = self.add_norm(add_norm_output, feed_forward_output, out_dense, self.out_dropout, self.out_layer_norm)
    return add_norm_output


'''
Lora-fied version of BertModel
'''
class LoraBertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([LoraBertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_dense_A = nn.Linear(config.hidden_size, LORA_SIZE)
    self.pooler_dense_B = nn.Linear(LORA_SIZE, config.hidden_size)
    self.pooler_dense.requires_grad = False

    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]
    # TODO
    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = self.word_embedding(input_ids)


    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)


    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    embeds_agr = inputs_embeds + pos_embeds + tk_type_embeds
    embeds_agr = self.embed_dropout(self.embed_layer_norm(embeds_agr))
    return embeds_agr


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Dora transform pooler dense
    pooler_dense = lambda x: self.pooler_dense(x) + self.pooler_dense_B(self.pooler_dense_A(x))
 
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}