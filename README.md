<a name="readme-top"></a>




<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<h3 align="center">Automatic Mouse Brain MRI Segmentation</h3>

  <p align="center">
    Automatic segmentation for mouse MRI. Applicable to four modalities: anatomical, DTI, NODDI, rsfMRI.
    <br />
    <a href="https://github.com/TheJacksonLaboratory/seg-for-4modalities/blob/main/user_guide_inference.pdf"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/TheJacksonLaboratory/seg-for-4modalities">View Demo</a>
    ·
    <a href="https://github.com/TheJacksonLaboratory/seg-for-4modalities/issues">Report Bug</a>
    ·
    <a href="https://github.com/TheJacksonLaboratory/seg-for-4modalities/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#useage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

A tool that leverages deep neural networks to perform skull stripping on mouse brain MRI images. Four modalities are supported: anatomical, diffusion tensor imaging (DTI), neurite orientation dispersion and density imaging (NODDI), and resting state ferromagnetic imaging (rsfMRI).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* conda - available at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Download the test data here, currently available on [Box](https://thejacksonlaboratory.box.com/s/doryp202qz1ar6800557p5t2d7w2nn6u)

2. Create a conda environment with the appropriate python version. The software was built using python==3.8
   ```sh
   conda create -y --name seg-for-4modalities python==3.8
   ```
3. Activate the environment
   ```sh
   conda activate seg-for-4modalities
   ```
4. Install the segmentation tool
   ```js
   pip install seg-for-4modalities
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

To check that the program is working correctly:
1. Navigate to the location the test dataest directory downloaded earlier was saved in a command line shell. In that directory, there should be two subdirectories: test_dataset and test_dataset_pregenerated.
2. Look through the pre-generated results included in the download (test_dataset_pregenerated)
4. Run segmentation on the clean test dataset
 ```sh
   python -m seg-for-4modalities.segment_brain --input_type dataset --input test_dataset

   ```
5. Check that all expected files were created in test_dataset. 
6. A shell script is included to remove segmentation files from the test dataset after processing to prepare for future iterations. 

_For more detailed information about installation, running the program, and detailed information about input parameters, please refer to the [Documentation](https://github.com/TheJacksonLaboratory/seg-for-4modalities/blob/main/user_guide_inference.pdf)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Zachary Frohock - zachary.frohock@jax.org

Project Link: [https://github.com/TheJacksonLaboratory/seg-for-4modalities](https://github.com/TheJacksonLaboratory/seg-for-4modalities)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TheJacksonLaboratory/seg-for-4modalities.svg?style=for-the-badge
[contributors-url]: https://github.com/TheJacksonLaboratory/seg-for-4modalities/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TheJacksonLaboratory/seg-for-4modalities.svg?style=for-the-badge
[forks-url]: https://github.com/TheJacksonLaboratory/seg-for-4modalities/network/members
[stars-shield]: https://img.shields.io/github/stars/TheJacksonLaboratory/seg-for-4modalities.svg?style=for-the-badge
[stars-url]: https://github.com/TheJacksonLaboratory/seg-for-4modalities/stargazers
[issues-shield]: https://img.shields.io/github/issues/TheJacksonLaboratory/seg-for-4modalities.svg?style=for-the-badge
[issues-url]: https://github.com/TheJacksonLaboratory/seg-for-4modalities/issues
[license-shield]: https://img.shields.io/github/license/TheJacksonLaboratory/seg-for-4modalities.svg?style=for-the-badge
[license-url]: https://github.com/TheJacksonLaboratory/seg-for-4modalities/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
