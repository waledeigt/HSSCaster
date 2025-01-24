# HSSCaster
Pipeline for novel method for obtaining normalised k-nu diagrams for various space weather monitoring


## Key notes on file types used
<ol>
  <li> <code>.json</code> files contain dictionary of information needed to setup f-mode masking and time selection of data. <b>NOTE:</b> <code>carr_rot</code> is selected t0 + 2 days time of analysis; <code>emerge_rot</code> is the NOAA time used for logging active region of interest </li>
  <li>all functions are in <code>ff_funcs.py</code></li>
  <li>an example script of how functions are utlised is also given <code>ff_tester.py</code>. User should change relevant file paths to accommodate their machine </li>
  <li>computed FF files (in heliographic coordinates) are in <code>CR20142015_hp_v4.zip</code></li> - either find in  via server or <url> https://doi.org/10.5281/zenodo.14732036 </url>
  
</ol>

## Full disk mapping

<ol>
  <li> Upadate Jan 2025: full disk normalised k-nu diagrams for full disk can be calculated using <code>norm_knu_FD</code> function </li>
  <li> Inputs: array of dates (any date format)</li>
  <li> Note: Initally, user needs to hardwire file locations for <code>CR20142015_hp_v4.zip</code> (line 541 in <code>ff_funcs.py</code>, <code>ff_path</code> variable) and data files (line 625 in <code>ff_funcs.py</code>)</li>
  </ol>

Example of code implentation:
    
  ```shell
    import ff_funcs as ptools

    normknu = ptools.norm_knu_FD(date=times) # where times is an array of date objects
  ```

One HMI 4-hr dopplergram file takes 30-35 seconds to produce full disk normalised k-nu diagrams (therefore 180-210 secs for 1 day, around 1.5 hours for one Carrington Rotation) 
  





  
