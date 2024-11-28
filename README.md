# HSSCaster
Pipeline for novel method for obtaining normalised k-nu diagrams for various space weather monitoring


## Key notes on file types used
<ol>
  <li> <code>.json</code> files contain dictionary of information needed to setup f-mode masking and time selection of data. <b>NOTE:</b> <code>carr_rot</code> is selected t0 + 2 days time of analysis; <code>emerge_rot</code> is the NOAA time used for logging active region of interest </li>
  <li>all functions are in <code>ff_funcs.py</code></li>
  <li>an example script of how functions are utlised is also given <code>ff_tester.py</code>. User should change relevant file paths to accommodate their machine </li>
  <li>computed FF files (in heliographic coordinates) are in <code>CR20142015_hp_v4.zip</code></li>
  
</ol>
