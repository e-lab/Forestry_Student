import numpy, math

def similarity(embed1, embed2):
    embed1 = numpy.array(embed1)
    embed2 = numpy.array(embed2)
    
    embed1_square = embed1.T @ embed1
    embed2_square = embed2.T @ embed2
    
    #return(math.sqrt(numpy.trace((embed1_square - embed2_square) @ (embed1_square - embed2_square).conj().T)))
    embed1_square = embed1_square.flatten()
    embed2_square = embed2_square.flatten()
    
    if len(embed1_square) != len(embed2_square):
        embed1_square = numpy.pad(embed1_square, (0, len(embed2_square) - len(embed1_square)))
    elif len(embed2_square) != len(embed1_square):
        embed2_square = numpy.pad(embed2_square, (0, len(embed1_square) - len(embed2_square)))
    
    return(
        embed1_square.dot(embed2_square) / (numpy.linalg.norm(embed1_square) * numpy.linalg.norm(embed2_square))
    )
    
    
