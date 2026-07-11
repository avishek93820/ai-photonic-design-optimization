def measure_diameter(binary_matrix):

    spans = []
    for row in binary_matrix:
        left = 0
        for idx, val in enumerate(row):
            if val != 0:
                left = idx
                break

        for idx, val in enumerate(row[::-1]):
            if val != 0:
                right = len(row) - idx
                spans.append(right - left)
                break
 
    return max(spans)

if __name__ == '__main__':
    mat = [[0,0,0,0,0],
           [0,0,1,1,1,0],
           [0,0,1,1,1,0,1,1,1,0,0,0,0],
           [0,1,1,1,0],
           [0,0,0,0,0],
           [0,0,0,0,0]]
    
    print(f"Diameter in pixels: {measure_diameter(mat)}")