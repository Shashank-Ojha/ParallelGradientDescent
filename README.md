# Parallelizing Gradient Descent on Different Architectures 
## Shashank Ojha and Kylee Santos


### Summary

We are going to create optimized implementations of gradient descent on both GPU and multi-core CPU platforms, and perform a detailed analysis of both systems’ performance characteristics. The GPU implementation will be done using CUDA, where as the multi-core CPU implementation will be done with OpenMP. 

### Background 



### Challenges

The challenges are more related to finding the best suited programming model given an architecture. This is because each architecture calls for a different implementation. With the high number of threads available to GPUs, we can potentially calculate the gradient with a larger number of points and make a more accurate step towards the optimal, but each update may be slower than on multi-core CPU's. Multi-core CPU's will have the advantage of making updates faster because there is no offload overhead involved. Clearly there is a tradeoff between correctness and speed and finding the programming model that optimizes performance in each case the main challenge. 

Another challenge of the project comes from the fact that we are working with large datasets, so how we organize our memory to hold all the data will be another area of focus for this project. For example, previous papers have stated that it may be beneficial to divide up the data into disjoint subsets so each core or block on the GPU doesn’t have to share data. Of course, this may lead to less accurate updates since each core or block only part of the whole picture. However, the speedup gained from less sharing maybe still allow us to arrive closer to the optimal sooner.

***Also talk about Batch vs Stochastic GD Tradeoffs***

### Resources

We will start off by implementing our gradient descent using CUDA on NVIDIA GeForce GTX 1080 GPUs and using OpenMP on Xeon Phi Machines. We will start the code from scratch since the actual implementation of the gradient descent isn't too complex and we may make several modifications to it based on the programming model. This assignment is an exploration of different programming models, so it does not make sense to build off someone else's code. We want full control of everything. 

There are several online papers about this topic. For now, we will use the following two papers for reference. 

1. http://martin.zinkevich.org/publications/nips2010.pdf

2. https://arxiv.org/pdf/1802.08800.pdf

The one thing we still have to figure out is how to obtain a good dataset. We want a data set that is large and minimize a non-convex function, so we can observe cases where our gradient descent might fall into a local minimum as opposed to a global one. 

If time permits, we might running our program on other machines and measuring performance on those as well. 

### Goals and Deliverables

### Platform Choice

### Schedule 

- Week of 11/5 - 11/9

- Week of 11/12 - 11/16

- Week of 11/19 - 11/23

- Week of 11/26 - 11/30

- Week of 12/3 - 12/7

- Week of 12/10 - 12/14



You can use the [editor on GitHub](https://github.com/Shashank-Ojha/ParallelGradientDescent/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Shashank-Ojha/ParallelGradientDescent/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
