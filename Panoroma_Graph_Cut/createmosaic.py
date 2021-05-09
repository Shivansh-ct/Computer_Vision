import cv2 as cv
import numpy as np
from graph_tool.all import *
import argparse


#wt_edge creates a weighted edge from g.vertex(i) to g.vertex(j)
def wt_edge(i,j):
    t = g.vertex(i)
    x = v_pixel[t][0]
    y = v_pixel[t][1]
    u = g.vertex(j)
    a = v_pixel[u][0]
    b = v_pixel[u][1]
    g.add_edge(t,u)
    e = g.edge(t,u)
    sim = np.array(np.sqrt(((1/3)*((p[x,y,0]-q[x,y,0])**2))+((1/3)*((p[x,y,1]-q[x,y,1])**2))+((1/3)*((p[x,y,2]-q[x,y,2])**2)))+np.sqrt(((1/3)*((p[a,b,0]-q[a,b,0])**2))+((1/3)*((p[a,b,1]-q[a,b,1])**2))+((1/3)*((p[a,b,2]-q[a,b,2])**2))),dtype=np.longdouble)
    gd = grad(x,y,a,b)
    if gd==0:
        edge_weights[e]=1000*count_ver*(count_ver-1)/2
    else:
        edge_weights[e] = sim/gd

#grad calculates the denominator of our edge cost function
def grad(x,y,a,b):
    return 2*np.array(np.linalg.norm(p[x,y,:]-p[a,b,:])+np.linalg.norm(q[x,y,:]-q[a,b,:]),dtype=np.longdouble)
  
if __name__ == "__main__":
    
    
     parser = argparse.ArgumentParser(description='Panorama creation')
     parser.add_argument('--input_path', help='Path to input image.folder')
     args = parser.parse_args()
     
         
     img_ = cv.imread(args.input_path+"img1.png")
     img_ = cv.resize(img_, (img_.shape[1]//(img_.shape[0]//500), 500))
     img = cv.imread(args.input_path+"img2.png")
     img = cv.resize(img, (img.shape[1]//(img.shape[0]//500), 500))
     img1 = cv.cvtColor(img_,cv.COLOR_BGR2GRAY)
     img2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
     
    #find the keypoints and descriptors with SIFT
     sift = cv.xfeatures2d.SIFT_create()
     
     kp1, des1 = sift.detectAndCompute(img2,None)
     kp2, des2 = sift.detectAndCompute(img1,None)
     
     bf = cv.BFMatcher()
     matches = bf.knnMatch(des1,des2, k=2)
     
     # Apply ratio test
     good = []
     for m in matches:
         if m[0].distance < 0.5*m[1].distance:
           good.append(m)
     matches = np.asarray(good)
     
     if len(matches[:,0]) >= 4:
         src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
         dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
         H_21, masked = cv.findHomography(src, dst, cv.RANSAC, 5.0)
     
     else:
         print("Panorama not possible")
         
     #Choosing among the images which is left or right, H_ij aligns image 'i' to image 'j' plane
     
     H_12 = np.linalg.inv(H_21)
     mn_12 = 0
     mn_21 = 0
     for y in range(img1.shape[0]):
       if np.matmul(H_12[0,0:3],np.transpose([0,y,1])) < mn_12:
           mn_12 = np.matmul(H_12[0,0:3],np.transpose([0,y,1]))
       if np.matmul(H_12[0,0:3],np.transpose([0,y,1])) < mn_21:
           mn_21 = np.matmul(H_21[0,0:3],np.transpose([0,y,1]))
     
     if mn_12<mn_21:
         right = cv.warpPerspective(img,H_21,(img.shape[1]+img_.shape[1],img.shape[0])) 
         left = np.zeros((img_.shape[0],img_.shape[1]+img.shape[1],img_.shape[2]),dtype=np.uint8)
         left[0:img_.shape[0],0:img_.shape[1],0:img_.shape[2]] = img_[:,:,:]
     else:
         right = cv.warpPerspective(img_,H_12,(img.shape[1]+img_.shape[1],img_.shape[0])) 
         left = np.zeros((img.shape[0],img_.shape[1]+img.shape[1],img.shape[2]),dtype=np.uint8)
         left[0:img.shape[0],0:img.shape[1],0:img.shape[2]] = img[:,:,:]
         x = img
         img = img_
         img_ = x
         x = img1
         img1 = img2
         img2 = x
         
                
     hp = cv.cvtColor(right,cv.COLOR_BGR2GRAY)
     mn = hp.shape[1]
     mo = 0
     for i in range(hp.shape[0]):
         for j in range(hp.shape[1]):
            if hp[i,j]!=0:
                if j<mn:
                    mn = j
                    break
         for j in range(hp.shape[1]):
            if hp[i,hp.shape[1]-j-1]!=0:
                if hp.shape[1]-j-1>mo:
                    mo = hp.shape[1]-j-1
                    break
     
     #mo stores the max column of right image that represents an image portion.
     #Finding Overlapping regions and storing the corresponding pixels in "Overlap" list.
     
     Overlap = []
     #count_ver stores the number of pixels in Overlapping region,i.e.,number of vertices
     count_ver=hp.shape[0]*(img1.shape[1]-mn-1)
     for i in range(hp.shape[0]):
         for j in range(mn+1,img1.shape[1],1):
                 Overlap.append([i,j])
     
     ver = img1.shape[0]
     hor = img1.shape[1]-mn-1
     x_max = img1.shape[1]-1
     x_min = mn+1
     y_max = img1.shape[0]-1
     y_min = 0
     
     
     #Making image value double for resolving overflow and precision problems
     p = np.array(left,dtype=np.longdouble)
     q = np.array(right,dtype=np.longdouble)
          
     #Initializing Graph for the Overlapping region
     g = Graph()
     #creating the vertex set
     V = g.add_vertex(count_ver+2)  #Last two vertices for source and destination
     
     #v_pixel maps from vertex to overlapping pixel,edge_weights stores the weight of an edge
     v_pixel = g.new_vertex_property("vector<int64_t>")
     edge_weights = g.new_edge_property("long double")
     for i in range(count_ver):
         v = g.vertex(i)
         v_pixel[v] = Overlap[i]
         
     
     #creating all the edges and assigning weights
     for i in range(count_ver):
         #print(i)
         if Overlap[i][1]<x_max and Overlap[i][1]>x_min and Overlap[i][0]<y_max and Overlap[i][0]>y_min:
            wt_edge(i,i-1)
            wt_edge(i,i+1)
            wt_edge(i,i-hor)
            wt_edge(i,i+hor)
            
         elif Overlap[i][1]==x_min and Overlap[i][0]==y_min:
             wt_edge(i,i+1)
             wt_edge(i,i+hor)
             
         elif Overlap[i][1]==x_min and Overlap[i][0]==y_max:
             wt_edge(i,i+1)
             wt_edge(i,i-hor)
             
         elif Overlap[i][1]==x_max and Overlap[i][0]==y_min:
             wt_edge(i,i-1)
             wt_edge(i,i+hor)
             
         elif Overlap[i][1]==x_max and Overlap[i][0]==y_max:
             wt_edge(i,i-1)
             wt_edge(i,i-hor)
           
         elif Overlap[i][1]==x_min:
            wt_edge(i,i+1)
            wt_edge(i,i+hor)
            wt_edge(i,i-hor)
           
         elif Overlap[i][1]==x_max:
            wt_edge(i,i-1)
            wt_edge(i,i+hor)
            wt_edge(i,i-hor)
          
         elif Overlap[i][0]==y_min:
            wt_edge(i,i-1)
            wt_edge(i,i+hor)
            wt_edge(i,i+1)
           
         elif Overlap[i][0]==y_max:
            wt_edge(i,i-1)
            wt_edge(i,i-hor)
            wt_edge(i,i+1)
          
         if Overlap[i][1]==x_min:
             s = g.vertex(count_ver)
             v = g.vertex(i)
             e = g.add_edge(s,v)
             e = g.edge(s,v)
             edge_weights[e] = 1000*count_ver*(count_ver-1)/2
             
         elif Overlap[i][1]==x_max:
             d = g.vertex(count_ver+1)
             v = g.vertex(i)
             e = g.add_edge(v,d)
             e = g.edge(v,d)
             edge_weights[e] = 1000*count_ver*(count_ver-1)/2
             
    
     s = g.vertex(count_ver)
     d = g.vertex(count_ver+1)
     res = g.new_edge_property("long double")
     
     #applying graph cut algorithm
     res = boykov_kolmogorov_max_flow(g, s, d, edge_weights)
     
     #part gives boolean values for vertices with true denoting the pixels belonging to left image
     part = min_st_cut(g, s, edge_weights, res)
    
     
     mask = np.zeros(right.shape,dtype=np.uint8)
     mask1 = np.zeros(right.shape,dtype=np.uint8)
     
     for i in range(count_ver):
          v = g.vertex(i)
          if part[v]==0:
             mask[v_pixel[v][0],v_pixel[v][1],:] = right[v_pixel[v][0],v_pixel[v][1],:]
             
          if part[v]==1:
             mask[v_pixel[v][0],v_pixel[v][1],:] = left[v_pixel[v][0],v_pixel[v][1],:]
             mask1[v_pixel[v][0],v_pixel[v][1],:] = np.array([1,1,1],dtype=np.uint8)
    
     
     mask[:,0:x_min,:]=left[:,0:x_min,:]
     mask1[:,0:x_min,:]=np.ones((mask1.shape[0],x_min,3),dtype=np.uint8)
     mask[:,x_max:,:]=right[:,x_max:,:]
     
     #Pyramid Blending with 4 levels
     
     #making the shape in power of 2 so that dividing in gaussian and laplacian indexing doesn't mess up
     u = np.array(np.power([2,2],np.ceil(np.log2([left.shape[0],left.shape[1]]))),dtype=np.int32)
     A = np.zeros((u[0],u[1],3),dtype=np.uint8)
     B = np.zeros((u[0],u[1],3),dtype=np.uint8)
     mask2 = np.zeros((u[0],u[1],3),dtype=np.uint8)
     
                 
     
     A[:left.shape[0],:left.shape[1],:]=left[:,:,:]
     B[:left.shape[0],:left.shape[1],:]=right[:,:,:]
     mask2[:left.shape[0],:left.shape[1],:]=mask1[:,:,:]
     
                 
     
     # generate Gaussian pyramid for A
     G = A.copy()
     gpA = [G]
     for i in range(4):
         G = cv.pyrDown(G)
         gpA.append(G)
         
     # generate Gaussian pyramid for B
     G = B.copy()
     gpB = [G]
     for i in range(4):
         G = cv.pyrDown(G)
         gpB.append(G)
     
     # generate Laplacian Pyramid for A
     
     lpA = [gpA[3]]
     for i in range(3,0,-1):
         GE = cv.pyrUp(gpA[i])
         L = cv.subtract(gpA[i-1],GE)
         lpA.append(L)
     
     # generate Laplacian Pyramid for B
     lpB = [gpB[3]]
     for i in range(3,0,-1):
         GE = cv.pyrUp(gpB[i])
         L = cv.subtract(gpB[i-1],GE)
         lpB.append(L)
         
     # Now add left and right halves of images in each level
     LS = []
     for la,lb in zip(lpA,lpB):
         mt = cv.resize(mask2,(la.shape[1],la.shape[0]))
         ls = np.zeros(la.shape,dtype=type(la[0,0,0]))
         scale = np.array(A.shape[0]/la.shape[0],dtype=np.int32)
         rows,cols,dpt = la.shape
         for i in range(rows):
           for j in range(cols):
             for k in range(dpt):
                 if mt[i,j,k]==1:
                     ls[i,j,k]=la[i,j,k]
                 else:
                     ls[i,j,k]=lb[i,j,k]
         LS.append(ls)
       
     
     # now reconstruct
     ls_ = LS[0]
     for i in range(1,4):
         ls_ = cv.pyrUp(ls_)
         ls_ = cv.add(ls_, LS[i])
     hello = np.array(ls_[0:left.shape[0],0:mo,:],dtype=np.uint8)
     
     #storing the final image
     cv.imwrite("final.png",hello)
    
    
    
    
    
