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

<h3 align="center">Automatic Mouse Brain MRI Segmentation</h3>

  <p align="center">
    Automatic segmentation for mouse MRI. Applicable to four modalities: anatomical, DTI, NODDI, rsfMRI. Associated with submission Frohock et al., Automated Artifact Removal and Segmentation of Mouse Brain Images Collected Using Multiple MRI Modalities
    <br />
    <a href="https://github.com/TheJacksonLaboratory/seg-for-4modalities/blob/inference_lightweight_package/user_guide_inference.pdf"><strong>Explore the docs »</strong></a>
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

* conda - available at [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Download and unzip the test data here, currently available on [Box](https://thejacksonlaboratory.box.com/s/7zciw09qfl7nc5phnnmvn49um7d9rh9m) 

2. Create a conda environment with the appropriate python version. The software was built using python==3.8
   ```sh
   conda create -y --name seg-for-4modalities python==3.8
   ```
3. Activate the environment
   ```sh
   conda activate seg-for-4modalities
   ```
4. Allow for scikit-learn installation as sklearn - required as of December 31st, 2023. To do so, set the environment variable SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True. The exact method to do so varies on the operating system you are using: \
   \
   For Windows CMD: 
   ```js 
   set SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
   ```  
   For Windows Powershell: 
   ```js 
   set SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL "True"
   ```  
   On MacOS or Linux Terminal: 
   ```js 
   export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
   ```

5. Install the segmentation tool
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

<!-- LIMITATIONS-->
## Known Issues

1. Installation on Macs with ARM architecture CPUs using pip fails. Recommend using intel Mac, Window, or a Linux distribution.
2. One or more dependencies have requirements referring to scikit-learn as sklearn. As of December 31, 2023, this is officially deprecated and causes installation to fail. As a workaround, it is possible to allow deprecated sklearn references to still install. To fix, set the environment variable: 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

See `LICENSE.txt` for more information.

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
