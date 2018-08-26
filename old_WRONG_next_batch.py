        while not input_queue.empty():     
            iteration = input_queue.get()
            print("Iteration: ", iteration) 
            current_index = iteration * batch_size 
            dist_row_list = []
            for i in tqdm(range(batch_size)):




                # begin the slice at the "current_index"-th row in
                # the first column
                slice_begin = [current_index, 0]
            
                # slice the embedding from "slice_begin" with shape
                # "slice_shape"
                slice_output = sess.run(slice_embedding, 
                                        feed_dict={
                                         SLICE_BEGIN:slice_begin
                                        }
                                       )
    
                # take dot product of slice with embedding
                dist_row = sess.run(mult, 
                                    feed_dict={
                                     SLICE_OUTPUT:slice_output
                                    }
                                   )                 
                sys.stdout.flush()

                dist_row_list.append(dist_row[0])
                current_index = current_index + 1
           
            # used to be doing this with tf.stack(), changing to numpy 
            # beacuse fuck that. 
            dist_matrix = np.stack(dist_row_list)
            sys.stdout.flush()
            output_queue.put(dist_matrix)
