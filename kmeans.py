import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    centers = [generator.randint(0, n)]
    for i in range(1, n_cluster):
        Distance = []
        for xxx in x:
            a = 2 ** 31
            for center in centers:
                a1 = np.linalg.norm(xxx - x[center]) ** 2
                a = min(a, a1)
            Distance.append(a)
        Distance = np.array(Distance)
        centers.append(np.argmax((Distance / np.sum(Distance)), axis=0))
    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)
class KMeans():
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)


        # centroids = np.zeros((self.n_cluster, D))
        def distanceeeeee(centroid, x, y):
            N = x.shape[0]
            l = []
            for xxx in range(self.n_cluster):
                l.append(np.sum((x[y == xxx] - centroid[xxx]) ** 2))
            l = np.array(l)
            return np.sum(l) / N

        y = np.zeros(N, dtype=int)
        centroids = x[self.centers]

        QQQ = distanceeeeee(centroids, x, y)

        i = 0
        while i < self.max_iter:
            
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2), axis=0)
            
            if np.absolute(QQQ - distanceeeeee(centroids, x, y)) <= self.e:
                break
            QQQ = distanceeeeee(centroids, x, y)
            ans=[]
            for xxx in range(self.n_cluster):
                

                a=np.mean(x[y==xxx], axis=0)
                ans.append(a)

            centroids = np.array(ans)
            i += 1

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, i


class KMeansClassifier():

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        #      'Implement fit function in KMeansClassifier class')

        
        centroids, membership, nums_of_updates = KMeans(self.n_cluster, self.max_iter, self.e).fit(x)
        votes=[]
        for i in range(self.n_cluster):
            votes.append({})
        a=[]
        for i in range(len(y)):
            a.append((y[i],membership[i]))
          
        for y, r in a:
            if y not in votes[r].keys():
                votes[r][y] = 1
            else:
                votes[r][y] += 1
        labels=[]
        for vote in votes:
            if not vote:
                labels.append(0)
            else:
                xx=max(vote, key=vote.get)
                labels.append(xx)
        labels=np.array(labels)
                
        centroid_labels = labels

        
        
        
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        
        
        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)


    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')
        
        labels = []
        for index in range(N):
            level=[]
            for n in range(self.n_cluster):
                a1=x[index,:]
                a2=self.centroids[n,:]
                a=np.subtract(a1,a2)
                a=np.inner(a,a)
                
                level.append(a)
            minn=np.argmin(level)
            labels.append(self.centroid_labels[minn])
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors
        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)
        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #          'Implement transform_image function')

    c = code_vectors.shape[0]

    m=image.shape[0]
    n=image.shape[1]
    

   
    
    new_im = np.zeros((m,n,3))
    
    for i in range(m):
        for j in range(n):
            level=[]
            for k in range(c):
                a1=image[i,j,:]
                a2=code_vectors[k,:]
                a=np.subtract(a1, a2)
                a=np.inner(a,a)
                level.append(a)
            
            
            minn=np.argmin(level)
            new_im[i,j,:] = code_vectors[minn,:]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im