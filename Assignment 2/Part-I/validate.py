##############
# Evaluation #
##############
#
# In evaluation, we simply feed the sequence and observe the output.
# The generation will be over once the "EOS" has been generated.
import torch
from utils.transformer import *
SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
def evaluate(encoder, decoder, bridge, input_tensor,device,index2word_hin, max_length=20,bidirectional=False):


    # Required for tensor matching.
    # Remove to see the results for educational purposes.
    with torch.no_grad():

        # Initialize the encoder hidden.
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(device)

        

        
        for ei in range(input_length):
            #Call the encoder for input tensors 
            encoder_output, encoder_hidden = #CODE_BLANK_1

            # only return the hidden and cell states for the last layer and pass it to the decoder
        #Store the encoder hidden cells in hn,cn
        hn, cn = #CODE_BLANK_2

        #Formatting the shape
        encoder_hn_last_layer = hn[-1].view(1,1,-1)
        encoder_cn_last_layer = cn[-1].view(1,1,-1)

        #Appending the hidden and the cell state
        encoder_hidden_last = [encoder_hn_last_layer, encoder_cn_last_layer]

        #Initialize the decoder input as SOS_token
        decoder_input =  #CODE_BLANK_3

        #Calling the bridge layers and storing it as decoder initial hidden inputs.
        encoder_hidden_last = [bridge(item) for item in encoder_hidden_last]
        decoder_hidden = encoder_hidden_last

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        #Iterate over maximum length of words
        for di in range(max_length):

            #Call the decoder object with hidden states and input as parameter
            decoder_output, decoder_hidden = #CODE_BLANK_4
            #Obtaining the top prediction amongst output which will be the output for that stage. This is called greedy approach
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                #CODE_BLANK_5(Append EOS token to the decoded word list.)
                break
            else:
                #Append the converted word from the tensor using appropriate dictionaries
                #CODE_BLANK_6
            #Squeezes the output index and will be fed again as the input for the next cell.
            decoder_input = topi.squeeze().detach()

        return decoded_words
         


######################################################################
def evaluateRandomly(encoder, decoder, bridge,device,testset,idx2word_en,idx2word_hin, n=10):
    j=0
    for i,data in enumerate(testset,1):
        j=j+1
        #Assign the value of data to pair
        pair =#CODE_BLANK_1

        #Getting the tensors and mask in the desired format
        input_tensor, mask_input = reformat_tensor_mask(pair[0].view(1,1,-1))
        
        #get the non zero values from the input tensor
        input_tensor = #CODE_BLANK_2
        
        #Getting the ground truth tensors in their desired format
        output_tensor, mask_output = reformat_tensor_mask(pair[1].view(1,1,-1))

        #Get values that are non zero from output-tensor
        output_tensor = #CODE_BLANK_3

        #Moving the tensors to gpu for faster processing.
        if device == torch.device("cuda"):
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

        #Call the SentenceFromTensor_ method using the idx2word dictionaries and the input and output tensor
        #You must join the words to form a sentence
        input_sentence = #CODE_BLANK_4
        output_sentence = #CODE_BLANK_5
        print('Input: ', input_sentence)
        print('Output: ', output_sentence)
        input_tensor=input_tensor.to(device)
        
        
        #CAlling the evaluate method
        output_words = #CODE_BlANK_6

        #Joining the predicted output to form the predicted sentence
        output_sentence = ' '.join(output_words)
        print('Predicted Output: ', output_sentence)
        print('')
        if(j==n):
          break

from preprocess import get_dataset
device = torch.device("cpu")
testset,idx2word_en,idx2word_hin = get_dataset(batch_size=1,types="val",shuffle=False,num_workers=1,pin_memory=False,drop_last=False)
encoder=torch.load("encoder.pt")
encoder=encoder.to(device)
decoder=torch.load("decoder.pt")
decoder=decoder.to(device)
bridge=torch.load("bridge.pt")
bridge=bridge.to(device)
evaluateRandomly(encoder,decoder,bridge,device,testset,idx2word_en,idx2word_hin)
 
