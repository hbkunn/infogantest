for m in range(-1,10):
    for n in range(-1,10):
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
        dets, scores, classes = detector.detect(image[max(m*100,0):min(300+m*100,1200),max(n*100,0):min(300+n*100,1200),::-1], 0.1)
        if dets.shape[0] == 0:
            continue
        runtime = t.toc()
        print('total spend: {}s'.format(runtime))
        lst.append(np.concatenate([(dets[:,0]+n*100).reshape(-1,1),(dets[:,1]+m*100).reshape(-1,1),(dets[:,2]+n*100).reshape(-1,1),
                                   (dets[:,3]+m*100).reshape(-1,1),scores.reshape(-1,1)],axis=1))
        count += dets.shape[0]

