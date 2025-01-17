# Simulating growth
The true strength of this package is in its ability to simulate cardiac growth. This notebook will guide you through the process of simulating cardiac growth using a canine model with mitral valve regurgitation as an example. We recommend to first familiarize yourself with the single beat simulations in the `single_beat.ipynb` notebook before proceeding.

## Background
### Mechanics-based growth models
We included several mechanics-based growth relationships based on reference [1-4]. The choice of growth model is up to the user and can be specified in the input file in `growth.type`. We use the kinematic growth model framework from Rodriguez et al. [5]. In brief, the way that all these models work is that they compare the current state of mechanics (typically elastic stretch) in each cardiac wall patch to that of a previous state. The growth model then calculates the amount of growth based on the difference between these states. The growth tensor *F_g* for each cardiac wall patch can be calculated using the following growth relationships:
1. `transverse_jones`: Transversely isotropic growth driven by changes in fiber (fiber/cross-fiber growth) and radial (radial growth) stretch based on Jones & Oomen [1].
2. `transverse_witzenburg`: Transversely isotropic growth driven by changes in fiber (fiber/cross-fiber growth) and radial (radial growth) stretch based on Witzenburg & Holmes [2].
3. `isotropic_oomen`: Isotropic growth driven by changes in fiber stretch based on Oomen et al. [3].
4. `isotropic_jones`: Isotropic growth driven by changes in fiber stretch based on Jones & Oomen [1].
5. `transverse_hybrid`: Combination of the transverse_jones and transverse_witzenburg growth models.

If you desire a different growth model, you can conveniently add your own growth model(s) in the `growth.py` file.

### Drivers of growth
The changes in mechanics that drive growth can be induced by a variety of pathologies or clinical interventions. The boundary conditions needed to simulate these can be specified in the `growth` section of the input file. The following drivers are implemented:
1. Systemic arterial resistance (`growth.ras`): to simulate changes in afterload e.g. due to aortic stenosis.
2. Mitral valve regurgitation (`growth.rmvb`): to simulate changes in preload due to mitral valve regurgitation.
3. Abnormal heart rate (`growth.hr`): to simulate changes in heart rate.
4. Blood volume (`growth.sbv`): to simulate changes in blood volume and/or vasoconstriction.
5. Myocardial ischemia (`growth.ischemic`): to simulate changes in contractility due to acute ischemia (no stiffening during chronic ischemia is implemented as of yet).
6. Dyssynchrony (`growth.t_act`): to simulate changes in activation timing such as in left bundle branch block and cardiac resynchronization therapy.
 
All these parameters will override the values specified in the other parameter sections (`heart`, `circulation`, `resistances`). The first 4 parameters need to be specified for each time point in the growth simulation, whereas changes in ischemia and dyssynchrony can be triggered at specific time points and need to be defined for each cardiac wall patch.


```python
# Jupyter magic
%load_ext autoreload
%autoreload 2
```

## Example: mitral valve regurgitation
We will here simulate transversely isotropic growth in a canine model with mitral valve regurgitation based on Jones & Oomen [1]. Just as in the single beat simulations, we need to specify the input file which contains the baseline model characteristics as well the growth boundary conditions. We use the `growth.rmvb` parameter to simulate mitral valve regurgitation and the growth model `transverse_jones` to simulate transversely isotropic growth. The model parameters are set to reflect a typical mitral regurgitation case in canines, see [1] for a more rigorous parameter calibration.

Just like in the single beat simulations, we first need to (1) initialize the model, (2) simulate growth, and (3) visualize the results. Start with model intitialization and print the baseline and acute mitral valve regurgitation parameters:


```python
from monarch import Hatch
import pathlib

# Initialize model
input_dir = pathlib.Path.cwd() / "input_files"
input_canine = input_dir / "canine_vo"
beats = Hatch(input_canine)

print("Growth type: " + beats.growth.type)
print("Baseline R_mvb: " + str(beats.growth.rmvb[0]))
print("Acute R_mvb: " + str(beats.growth.rmvb[1]))
```

    Growth type: transverse_jones
    Baseline R_mvb: 10000000000.0
    Acute R_mvb: 0.3


A lower resistance means there is more regurgitation, with the intitial value set sufficiently high to ensure no regurgitation. We can now simulate growth , note that just like in the single beat simulations, the first simulation you run may take a while as the model is compiled. Subsequent simulations will be faster, rerun the cell to see this for yourself.


```python
# Simulate growth
beats.let_it_grow()
```

All model readouts are included in a Pandas format table `beat.growth.outputs`, similar to the single beat `beat.outputs` table but now every row includes a growth time point, with the index indicating which time point (in days).


```python
beats.growth.outputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LVEDV</th>
      <th>LVESV</th>
      <th>LVEDP</th>
      <th>LVESP</th>
      <th>LVMaxP</th>
      <th>LVMaxdP</th>
      <th>LVMindP</th>
      <th>LVSV</th>
      <th>LVRF</th>
      <th>LVEF</th>
      <th>...</th>
      <th>RVESVi</th>
      <th>dLVEDV</th>
      <th>dLVESV</th>
      <th>dLVEF</th>
      <th>dEDWthLfw</th>
      <th>dESWthLfw</th>
      <th>dEDWthRfw</th>
      <th>dESWthRfw</th>
      <th>dEDWthSw</th>
      <th>dESWthSw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>75.754258</td>
      <td>28.880644</td>
      <td>14.210911</td>
      <td>89.694544</td>
      <td>115.346887</td>
      <td>1850.201504</td>
      <td>-1242.582740</td>
      <td>46.873614</td>
      <td>4.254705e-11</td>
      <td>0.618759</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>83.476233</td>
      <td>18.974578</td>
      <td>20.307621</td>
      <td>64.849933</td>
      <td>83.574090</td>
      <td>1171.359824</td>
      <td>-866.988531</td>
      <td>64.501655</td>
      <td>5.062669e-01</td>
      <td>0.772695</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.101935</td>
      <td>-0.343000</td>
      <td>0.248782</td>
      <td>-0.035321</td>
      <td>0.087262</td>
      <td>0.028475</td>
      <td>-0.011606</td>
      <td>-0.050763</td>
      <td>0.139324</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92.421542</td>
      <td>26.951722</td>
      <td>19.772233</td>
      <td>65.425740</td>
      <td>84.193418</td>
      <td>1146.308269</td>
      <td>-847.412375</td>
      <td>65.469820</td>
      <td>5.119206e-01</td>
      <td>0.708383</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.220018</td>
      <td>-0.066789</td>
      <td>0.144845</td>
      <td>-0.133543</td>
      <td>-0.051496</td>
      <td>0.012284</td>
      <td>-0.024406</td>
      <td>-0.156732</td>
      <td>-0.002725</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98.218477</td>
      <td>31.161504</td>
      <td>19.167024</td>
      <td>66.342824</td>
      <td>85.704264</td>
      <td>1166.910949</td>
      <td>-854.829746</td>
      <td>67.056973</td>
      <td>5.176848e-01</td>
      <td>0.682733</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.296541</td>
      <td>0.078975</td>
      <td>0.103391</td>
      <td>-0.152722</td>
      <td>-0.087082</td>
      <td>-0.000451</td>
      <td>-0.030215</td>
      <td>-0.178855</td>
      <td>-0.041457</td>
    </tr>
    <tr>
      <th>6</th>
      <td>103.288372</td>
      <td>34.790691</td>
      <td>18.507365</td>
      <td>67.274366</td>
      <td>87.128514</td>
      <td>1186.295067</td>
      <td>-864.063249</td>
      <td>68.497681</td>
      <td>5.224598e-01</td>
      <td>0.663169</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.363466</td>
      <td>0.204637</td>
      <td>0.071774</td>
      <td>-0.160760</td>
      <td>-0.107582</td>
      <td>-0.007954</td>
      <td>-0.031938</td>
      <td>-0.190842</td>
      <td>-0.066707</td>
    </tr>
    <tr>
      <th>8</th>
      <td>107.768485</td>
      <td>38.044233</td>
      <td>18.080916</td>
      <td>68.271677</td>
      <td>88.413252</td>
      <td>1201.819414</td>
      <td>-873.125672</td>
      <td>69.724252</td>
      <td>5.263717e-01</td>
      <td>0.646982</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.422606</td>
      <td>0.317292</td>
      <td>0.045612</td>
      <td>-0.163789</td>
      <td>-0.120907</td>
      <td>-0.011938</td>
      <td>-0.031116</td>
      <td>-0.197124</td>
      <td>-0.084327</td>
    </tr>
    <tr>
      <th>10</th>
      <td>111.728671</td>
      <td>40.916996</td>
      <td>17.591236</td>
      <td>69.082592</td>
      <td>89.553025</td>
      <td>1218.378218</td>
      <td>-881.864066</td>
      <td>70.811675</td>
      <td>5.295804e-01</td>
      <td>0.633782</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.474883</td>
      <td>0.416762</td>
      <td>0.024280</td>
      <td>-0.163534</td>
      <td>-0.128860</td>
      <td>-0.015086</td>
      <td>-0.030498</td>
      <td>-0.200583</td>
      <td>-0.097556</td>
    </tr>
    <tr>
      <th>12</th>
      <td>115.253790</td>
      <td>43.495173</td>
      <td>17.173738</td>
      <td>69.730066</td>
      <td>90.574591</td>
      <td>1231.075413</td>
      <td>-889.586425</td>
      <td>71.758618</td>
      <td>5.322129e-01</td>
      <td>0.622614</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.521417</td>
      <td>0.506032</td>
      <td>0.006230</td>
      <td>-0.162074</td>
      <td>-0.134141</td>
      <td>-0.017609</td>
      <td>-0.030160</td>
      <td>-0.202187</td>
      <td>-0.107876</td>
    </tr>
    <tr>
      <th>14</th>
      <td>118.428528</td>
      <td>45.857180</td>
      <td>16.822122</td>
      <td>70.237259</td>
      <td>91.412376</td>
      <td>1244.504010</td>
      <td>-896.003925</td>
      <td>72.571348</td>
      <td>5.343805e-01</td>
      <td>0.612786</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.563325</td>
      <td>0.587817</td>
      <td>-0.009653</td>
      <td>-0.160718</td>
      <td>-0.138578</td>
      <td>-0.019607</td>
      <td>-0.030062</td>
      <td>-0.202596</td>
      <td>-0.116152</td>
    </tr>
    <tr>
      <th>16</th>
      <td>121.300032</td>
      <td>48.024877</td>
      <td>16.529092</td>
      <td>70.627213</td>
      <td>92.117001</td>
      <td>1251.487784</td>
      <td>-901.477914</td>
      <td>73.275155</td>
      <td>5.361730e-01</td>
      <td>0.604082</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.601231</td>
      <td>0.662874</td>
      <td>-0.023720</td>
      <td>-0.159468</td>
      <td>-0.142325</td>
      <td>-0.021172</td>
      <td>-0.030156</td>
      <td>-0.202217</td>
      <td>-0.122927</td>
    </tr>
    <tr>
      <th>18</th>
      <td>123.913733</td>
      <td>50.107593</td>
      <td>16.286132</td>
      <td>71.236017</td>
      <td>92.747684</td>
      <td>1260.591680</td>
      <td>-906.080251</td>
      <td>73.806140</td>
      <td>5.376519e-01</td>
      <td>0.595625</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.635733</td>
      <td>0.734989</td>
      <td>-0.037387</td>
      <td>-0.158307</td>
      <td>-0.146520</td>
      <td>-0.022027</td>
      <td>-0.030330</td>
      <td>-0.201361</td>
      <td>-0.128217</td>
    </tr>
    <tr>
      <th>20</th>
      <td>126.310161</td>
      <td>51.944072</td>
      <td>16.092594</td>
      <td>71.464632</td>
      <td>93.301641</td>
      <td>1267.221948</td>
      <td>-909.947300</td>
      <td>74.366089</td>
      <td>5.387804e-01</td>
      <td>0.588758</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.667367</td>
      <td>0.798577</td>
      <td>-0.048486</td>
      <td>-0.157258</td>
      <td>-0.149285</td>
      <td>-0.022286</td>
      <td>-0.031145</td>
      <td>-0.200153</td>
      <td>-0.132840</td>
    </tr>
    <tr>
      <th>22</th>
      <td>128.506076</td>
      <td>53.735416</td>
      <td>15.938778</td>
      <td>71.947458</td>
      <td>93.797274</td>
      <td>1272.175744</td>
      <td>-913.470549</td>
      <td>74.770661</td>
      <td>5.398286e-01</td>
      <td>0.581845</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.696354</td>
      <td>0.860603</td>
      <td>-0.059657</td>
      <td>-0.156296</td>
      <td>-0.152650</td>
      <td>-0.022309</td>
      <td>-0.031660</td>
      <td>-0.198611</td>
      <td>-0.136352</td>
    </tr>
    <tr>
      <th>24</th>
      <td>130.514664</td>
      <td>55.295124</td>
      <td>15.814535</td>
      <td>72.042146</td>
      <td>94.216994</td>
      <td>1275.957326</td>
      <td>-916.228436</td>
      <td>75.219540</td>
      <td>5.405958e-01</td>
      <td>0.576330</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.722869</td>
      <td>0.914608</td>
      <td>-0.068571</td>
      <td>-0.155387</td>
      <td>-0.154639</td>
      <td>-0.022100</td>
      <td>-0.032615</td>
      <td>-0.196751</td>
      <td>-0.139461</td>
    </tr>
    <tr>
      <th>26</th>
      <td>132.373329</td>
      <td>56.844659</td>
      <td>15.535226</td>
      <td>72.420310</td>
      <td>94.602313</td>
      <td>1278.601178</td>
      <td>-918.861370</td>
      <td>75.528669</td>
      <td>5.413688e-01</td>
      <td>0.570573</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.747404</td>
      <td>0.968261</td>
      <td>-0.077875</td>
      <td>-0.154352</td>
      <td>-0.157338</td>
      <td>-0.021959</td>
      <td>-0.033216</td>
      <td>-0.195291</td>
      <td>-0.141979</td>
    </tr>
    <tr>
      <th>28</th>
      <td>134.095349</td>
      <td>58.204938</td>
      <td>15.468306</td>
      <td>72.441609</td>
      <td>94.947181</td>
      <td>1280.209238</td>
      <td>-921.179505</td>
      <td>75.890411</td>
      <td>5.420644e-01</td>
      <td>0.565944</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.770136</td>
      <td>1.015361</td>
      <td>-0.085357</td>
      <td>-0.153595</td>
      <td>-0.158843</td>
      <td>-0.021586</td>
      <td>-0.034068</td>
      <td>-0.193594</td>
      <td>-0.144683</td>
    </tr>
    <tr>
      <th>30</th>
      <td>135.692973</td>
      <td>59.565066</td>
      <td>15.426425</td>
      <td>72.746402</td>
      <td>95.254117</td>
      <td>1282.908571</td>
      <td>-923.154262</td>
      <td>76.127907</td>
      <td>5.426830e-01</td>
      <td>0.561031</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.791226</td>
      <td>1.062456</td>
      <td>-0.093297</td>
      <td>-0.152899</td>
      <td>-0.161050</td>
      <td>-0.021170</td>
      <td>-0.034590</td>
      <td>-0.191964</td>
      <td>-0.146830</td>
    </tr>
    <tr>
      <th>32</th>
      <td>137.178077</td>
      <td>60.840979</td>
      <td>15.208641</td>
      <td>73.020053</td>
      <td>95.528795</td>
      <td>1287.370550</td>
      <td>-924.974907</td>
      <td>76.337098</td>
      <td>5.432343e-01</td>
      <td>0.556482</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.810830</td>
      <td>1.106635</td>
      <td>-0.100648</td>
      <td>-0.152035</td>
      <td>-0.163049</td>
      <td>-0.020930</td>
      <td>-0.035079</td>
      <td>-0.190754</td>
      <td>-0.148814</td>
    </tr>
    <tr>
      <th>34</th>
      <td>138.561579</td>
      <td>61.946111</td>
      <td>15.206101</td>
      <td>72.942081</td>
      <td>95.775011</td>
      <td>1286.656657</td>
      <td>-926.580465</td>
      <td>76.615469</td>
      <td>5.437247e-01</td>
      <td>0.552934</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.829093</td>
      <td>1.144901</td>
      <td>-0.106381</td>
      <td>-0.151449</td>
      <td>-0.163958</td>
      <td>-0.020482</td>
      <td>-0.035842</td>
      <td>-0.189255</td>
      <td>-0.151041</td>
    </tr>
    <tr>
      <th>36</th>
      <td>139.853088</td>
      <td>63.072228</td>
      <td>15.020370</td>
      <td>73.167249</td>
      <td>95.997019</td>
      <td>1289.302570</td>
      <td>-927.946155</td>
      <td>76.780861</td>
      <td>5.441621e-01</td>
      <td>0.549011</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.846142</td>
      <td>1.183893</td>
      <td>-0.112722</td>
      <td>-0.150685</td>
      <td>-0.165622</td>
      <td>-0.020244</td>
      <td>-0.036269</td>
      <td>-0.188181</td>
      <td>-0.152745</td>
    </tr>
    <tr>
      <th>38</th>
      <td>141.061242</td>
      <td>64.132511</td>
      <td>14.849070</td>
      <td>73.370973</td>
      <td>96.198400</td>
      <td>1290.023764</td>
      <td>-929.243646</td>
      <td>76.928731</td>
      <td>5.445542e-01</td>
      <td>0.545357</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.862090</td>
      <td>1.220605</td>
      <td>-0.118628</td>
      <td>-0.149968</td>
      <td>-0.167147</td>
      <td>-0.020018</td>
      <td>-0.036663</td>
      <td>-0.187170</td>
      <td>-0.154328</td>
    </tr>
    <tr>
      <th>40</th>
      <td>142.193318</td>
      <td>65.132181</td>
      <td>14.892559</td>
      <td>73.557342</td>
      <td>96.380975</td>
      <td>1291.457732</td>
      <td>-930.405204</td>
      <td>77.061137</td>
      <td>5.449085e-01</td>
      <td>0.541946</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.877034</td>
      <td>1.255219</td>
      <td>-0.124140</td>
      <td>-0.149528</td>
      <td>-0.168548</td>
      <td>-0.019588</td>
      <td>-0.037026</td>
      <td>-0.185842</td>
      <td>-0.155802</td>
    </tr>
    <tr>
      <th>42</th>
      <td>143.256383</td>
      <td>66.075888</td>
      <td>14.745413</td>
      <td>73.728592</td>
      <td>96.547663</td>
      <td>1291.515553</td>
      <td>-931.451110</td>
      <td>77.180495</td>
      <td>5.452287e-01</td>
      <td>0.538758</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.891067</td>
      <td>1.287895</td>
      <td>-0.129293</td>
      <td>-0.148896</td>
      <td>-0.169840</td>
      <td>-0.019393</td>
      <td>-0.037359</td>
      <td>-0.184942</td>
      <td>-0.157178</td>
    </tr>
    <tr>
      <th>44</th>
      <td>144.255870</td>
      <td>66.872907</td>
      <td>14.608864</td>
      <td>73.557216</td>
      <td>96.700250</td>
      <td>1293.879090</td>
      <td>-932.377145</td>
      <td>77.382963</td>
      <td>5.455195e-01</td>
      <td>0.536429</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.904261</td>
      <td>1.315492</td>
      <td>-0.133057</td>
      <td>-0.148301</td>
      <td>-0.170151</td>
      <td>-0.019217</td>
      <td>-0.037981</td>
      <td>-0.184092</td>
      <td>-0.158873</td>
    </tr>
    <tr>
      <th>46</th>
      <td>145.196717</td>
      <td>67.716141</td>
      <td>14.687886</td>
      <td>73.702050</td>
      <td>96.839719</td>
      <td>1291.782917</td>
      <td>-933.213659</td>
      <td>77.480576</td>
      <td>5.457839e-01</td>
      <td>0.533625</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.916681</td>
      <td>1.344689</td>
      <td>-0.137588</td>
      <td>-0.147981</td>
      <td>-0.171259</td>
      <td>-0.018826</td>
      <td>-0.038259</td>
      <td>-0.182896</td>
      <td>-0.160080</td>
    </tr>
    <tr>
      <th>48</th>
      <td>146.083994</td>
      <td>68.514791</td>
      <td>14.569136</td>
      <td>73.835878</td>
      <td>96.967607</td>
      <td>1294.602251</td>
      <td>-934.007145</td>
      <td>77.569203</td>
      <td>5.460270e-01</td>
      <td>0.530990</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.928393</td>
      <td>1.372343</td>
      <td>-0.141846</td>
      <td>-0.147454</td>
      <td>-0.172287</td>
      <td>-0.018685</td>
      <td>-0.038512</td>
      <td>-0.182132</td>
      <td>-0.161213</td>
    </tr>
    <tr>
      <th>50</th>
      <td>146.921348</td>
      <td>69.271795</td>
      <td>14.458275</td>
      <td>73.959718</td>
      <td>97.085472</td>
      <td>1294.613767</td>
      <td>-934.728019</td>
      <td>77.649553</td>
      <td>5.462510e-01</td>
      <td>0.528511</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.939447</td>
      <td>1.398554</td>
      <td>-0.145853</td>
      <td>-0.146957</td>
      <td>-0.173241</td>
      <td>-0.018559</td>
      <td>-0.038741</td>
      <td>-0.181407</td>
      <td>-0.162278</td>
    </tr>
    <tr>
      <th>52</th>
      <td>147.684976</td>
      <td>69.964998</td>
      <td>14.554058</td>
      <td>74.038761</td>
      <td>97.160041</td>
      <td>1292.671021</td>
      <td>-935.084590</td>
      <td>77.719978</td>
      <td>5.463075e-01</td>
      <td>0.526255</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.949527</td>
      <td>1.422557</td>
      <td>-0.149499</td>
      <td>-0.146664</td>
      <td>-0.173998</td>
      <td>-0.018193</td>
      <td>-0.039034</td>
      <td>-0.180219</td>
      <td>-0.163207</td>
    </tr>
    <tr>
      <th>54</th>
      <td>148.445394</td>
      <td>70.659550</td>
      <td>14.464175</td>
      <td>74.170339</td>
      <td>97.283555</td>
      <td>1295.208211</td>
      <td>-935.894821</td>
      <td>77.785844</td>
      <td>5.466164e-01</td>
      <td>0.524003</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.959565</td>
      <td>1.446606</td>
      <td>-0.153138</td>
      <td>-0.146266</td>
      <td>-0.174914</td>
      <td>-0.018096</td>
      <td>-0.039180</td>
      <td>-0.179629</td>
      <td>-0.164196</td>
    </tr>
    <tr>
      <th>56</th>
      <td>149.131428</td>
      <td>71.190449</td>
      <td>14.364788</td>
      <td>73.907270</td>
      <td>97.349127</td>
      <td>1295.825692</td>
      <td>-936.203725</td>
      <td>77.940978</td>
      <td>5.466713e-01</td>
      <td>0.522633</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.968621</td>
      <td>1.464988</td>
      <td>-0.155353</td>
      <td>-0.145796</td>
      <td>-0.174726</td>
      <td>-0.018001</td>
      <td>-0.039745</td>
      <td>-0.178935</td>
      <td>-0.165452</td>
    </tr>
    <tr>
      <th>58</th>
      <td>149.819395</td>
      <td>71.820997</td>
      <td>14.288192</td>
      <td>74.028086</td>
      <td>97.460398</td>
      <td>1294.297507</td>
      <td>-936.907701</td>
      <td>77.998398</td>
      <td>5.469476e-01</td>
      <td>0.520616</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.977703</td>
      <td>1.486821</td>
      <td>-0.158612</td>
      <td>-0.145454</td>
      <td>-0.175550</td>
      <td>-0.017934</td>
      <td>-0.039846</td>
      <td>-0.178422</td>
      <td>-0.166345</td>
    </tr>
    <tr>
      <th>60</th>
      <td>150.435910</td>
      <td>72.388287</td>
      <td>14.414720</td>
      <td>74.089181</td>
      <td>97.515895</td>
      <td>1293.703427</td>
      <td>-937.168323</td>
      <td>78.047623</td>
      <td>5.469927e-01</td>
      <td>0.518810</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.985841</td>
      <td>1.506464</td>
      <td>-0.161531</td>
      <td>-0.145280</td>
      <td>-0.176137</td>
      <td>-0.017594</td>
      <td>-0.040046</td>
      <td>-0.177374</td>
      <td>-0.167093</td>
    </tr>
    <tr>
      <th>62</th>
      <td>151.023442</td>
      <td>72.933196</td>
      <td>14.333883</td>
      <td>74.156795</td>
      <td>97.577252</td>
      <td>1295.261605</td>
      <td>-937.499349</td>
      <td>78.090246</td>
      <td>5.471029e-01</td>
      <td>0.517074</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.993597</td>
      <td>1.525331</td>
      <td>-0.164337</td>
      <td>-0.144893</td>
      <td>-0.176724</td>
      <td>-0.017512</td>
      <td>-0.040218</td>
      <td>-0.176800</td>
      <td>-0.167823</td>
    </tr>
    <tr>
      <th>64</th>
      <td>151.621612</td>
      <td>73.483790</td>
      <td>14.273366</td>
      <td>74.266613</td>
      <td>97.678276</td>
      <td>1296.260164</td>
      <td>-938.186309</td>
      <td>78.137823</td>
      <td>5.473612e-01</td>
      <td>0.515348</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.001493</td>
      <td>1.544396</td>
      <td>-0.167127</td>
      <td>-0.144625</td>
      <td>-0.177446</td>
      <td>-0.017430</td>
      <td>-0.040280</td>
      <td>-0.176402</td>
      <td>-0.168599</td>
    </tr>
    <tr>
      <th>66</th>
      <td>152.151722</td>
      <td>73.975699</td>
      <td>14.199183</td>
      <td>74.317383</td>
      <td>97.722875</td>
      <td>1294.805696</td>
      <td>-938.386545</td>
      <td>78.176023</td>
      <td>5.474006e-01</td>
      <td>0.513803</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.008491</td>
      <td>1.561428</td>
      <td>-0.169623</td>
      <td>-0.144262</td>
      <td>-0.177935</td>
      <td>-0.017337</td>
      <td>-0.040445</td>
      <td>-0.175869</td>
      <td>-0.169242</td>
    </tr>
    <tr>
      <th>68</th>
      <td>152.657626</td>
      <td>74.448653</td>
      <td>14.131063</td>
      <td>74.373463</td>
      <td>97.772204</td>
      <td>1293.424337</td>
      <td>-938.643755</td>
      <td>78.208973</td>
      <td>5.474921e-01</td>
      <td>0.512316</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.015169</td>
      <td>1.577804</td>
      <td>-0.172026</td>
      <td>-0.143928</td>
      <td>-0.178426</td>
      <td>-0.017242</td>
      <td>-0.040591</td>
      <td>-0.175381</td>
      <td>-0.169868</td>
    </tr>
    <tr>
      <th>70</th>
      <td>153.142008</td>
      <td>74.901817</td>
      <td>14.068352</td>
      <td>74.430769</td>
      <td>97.822442</td>
      <td>1294.071998</td>
      <td>-938.917557</td>
      <td>78.240191</td>
      <td>5.475919e-01</td>
      <td>0.510900</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.021563</td>
      <td>1.593495</td>
      <td>-0.174315</td>
      <td>-0.143620</td>
      <td>-0.178904</td>
      <td>-0.017142</td>
      <td>-0.040722</td>
      <td>-0.174935</td>
      <td>-0.170470</td>
    </tr>
    <tr>
      <th>72</th>
      <td>153.605627</td>
      <td>75.335706</td>
      <td>14.010082</td>
      <td>74.487464</td>
      <td>97.871949</td>
      <td>1295.260361</td>
      <td>-939.191999</td>
      <td>78.269921</td>
      <td>5.476907e-01</td>
      <td>0.509551</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.027683</td>
      <td>1.608519</td>
      <td>-0.176495</td>
      <td>-0.143334</td>
      <td>-0.179365</td>
      <td>-0.017038</td>
      <td>-0.040841</td>
      <td>-0.174524</td>
      <td>-0.171047</td>
    </tr>
    <tr>
      <th>74</th>
      <td>154.049206</td>
      <td>75.751126</td>
      <td>14.173511</td>
      <td>74.542665</td>
      <td>97.920196</td>
      <td>1295.976239</td>
      <td>-939.459773</td>
      <td>78.298080</td>
      <td>5.477869e-01</td>
      <td>0.508267</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.033539</td>
      <td>1.622903</td>
      <td>-0.178571</td>
      <td>-0.143322</td>
      <td>-0.179806</td>
      <td>-0.016662</td>
      <td>-0.040952</td>
      <td>-0.173703</td>
      <td>-0.171598</td>
    </tr>
    <tr>
      <th>76</th>
      <td>154.473564</td>
      <td>76.051979</td>
      <td>14.121818</td>
      <td>74.256757</td>
      <td>97.966370</td>
      <td>1294.825940</td>
      <td>-939.714745</td>
      <td>78.421585</td>
      <td>5.478775e-01</td>
      <td>0.507670</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.039140</td>
      <td>1.633320</td>
      <td>-0.179535</td>
      <td>-0.143068</td>
      <td>-0.179396</td>
      <td>-0.016553</td>
      <td>-0.041396</td>
      <td>-0.173343</td>
      <td>-0.172564</td>
    </tr>
    <tr>
      <th>78</th>
      <td>154.879532</td>
      <td>76.432746</td>
      <td>14.073042</td>
      <td>74.308631</td>
      <td>98.010360</td>
      <td>1293.715358</td>
      <td>-939.953807</td>
      <td>78.446786</td>
      <td>5.479632e-01</td>
      <td>0.506502</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.044499</td>
      <td>1.646504</td>
      <td>-0.181423</td>
      <td>-0.142828</td>
      <td>-0.179799</td>
      <td>-0.016444</td>
      <td>-0.041494</td>
      <td>-0.173004</td>
      <td>-0.173069</td>
    </tr>
    <tr>
      <th>80</th>
      <td>155.267991</td>
      <td>76.797482</td>
      <td>14.026836</td>
      <td>74.357401</td>
      <td>98.052101</td>
      <td>1292.642344</td>
      <td>-940.169662</td>
      <td>78.470510</td>
      <td>5.480445e-01</td>
      <td>0.505388</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.049627</td>
      <td>1.659133</td>
      <td>-0.183224</td>
      <td>-0.142599</td>
      <td>-0.180183</td>
      <td>-0.016334</td>
      <td>-0.041587</td>
      <td>-0.172684</td>
      <td>-0.173551</td>
    </tr>
    <tr>
      <th>82</th>
      <td>155.639700</td>
      <td>77.147015</td>
      <td>13.982920</td>
      <td>74.403816</td>
      <td>98.091605</td>
      <td>1293.253921</td>
      <td>-940.397965</td>
      <td>78.492685</td>
      <td>5.481219e-01</td>
      <td>0.504323</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.054534</td>
      <td>1.671236</td>
      <td>-0.184944</td>
      <td>-0.142381</td>
      <td>-0.180548</td>
      <td>-0.016222</td>
      <td>-0.041675</td>
      <td>-0.172380</td>
      <td>-0.174012</td>
    </tr>
    <tr>
      <th>84</th>
      <td>155.995536</td>
      <td>77.482082</td>
      <td>13.941157</td>
      <td>74.448005</td>
      <td>98.128977</td>
      <td>1294.094200</td>
      <td>-940.611488</td>
      <td>78.513454</td>
      <td>5.481952e-01</td>
      <td>0.503306</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.059231</td>
      <td>1.682838</td>
      <td>-0.186588</td>
      <td>-0.142172</td>
      <td>-0.180895</td>
      <td>-0.016109</td>
      <td>-0.041759</td>
      <td>-0.172093</td>
      <td>-0.174454</td>
    </tr>
    <tr>
      <th>86</th>
      <td>156.336241</td>
      <td>77.803367</td>
      <td>13.901375</td>
      <td>74.490132</td>
      <td>98.164305</td>
      <td>1294.872223</td>
      <td>-940.812469</td>
      <td>78.532874</td>
      <td>5.482646e-01</td>
      <td>0.502333</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.063729</td>
      <td>1.693962</td>
      <td>-0.188160</td>
      <td>-0.141972</td>
      <td>-0.181225</td>
      <td>-0.015996</td>
      <td>-0.041839</td>
      <td>-0.171820</td>
      <td>-0.174876</td>
    </tr>
    <tr>
      <th>88</th>
      <td>156.662563</td>
      <td>78.111516</td>
      <td>13.863449</td>
      <td>74.530291</td>
      <td>98.197706</td>
      <td>1294.871414</td>
      <td>-941.001760</td>
      <td>78.551047</td>
      <td>5.483308e-01</td>
      <td>0.501403</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.068036</td>
      <td>1.704632</td>
      <td>-0.189664</td>
      <td>-0.141781</td>
      <td>-0.181539</td>
      <td>-0.015882</td>
      <td>-0.041916</td>
      <td>-0.171561</td>
      <td>-0.175281</td>
    </tr>
    <tr>
      <th>90</th>
      <td>156.975205</td>
      <td>78.407126</td>
      <td>13.827278</td>
      <td>74.568591</td>
      <td>98.229328</td>
      <td>1293.966448</td>
      <td>-941.180167</td>
      <td>78.568078</td>
      <td>5.483923e-01</td>
      <td>0.500513</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.072163</td>
      <td>1.714868</td>
      <td>-0.191102</td>
      <td>-0.141597</td>
      <td>-0.181838</td>
      <td>-0.015767</td>
      <td>-0.041990</td>
      <td>-0.171315</td>
      <td>-0.175669</td>
    </tr>
  </tbody>
</table>
<p>47 rows × 91 columns</p>
</div>



For easier analysis, any of the growth outputs can be plotted over time using the `plot_growth` function. The function takes a list of outputs (and optional units for plotting purposes) to plot. Here we plot the left ventricular end-diastolic volume (LVEDV) and end-systolic volume (LVESV):


```python
import monarch.metamophoses as meta

meta.plot_growth(beats, ["LVEDV", "LVESV"], units=["mL", "mL"])
```


    
![png](growth_files/growth_9_0.png)
    


One can also plot the pressure-volume loops of a single compartment over time. Here we plot the pressure-volume loop of the left ventricle at all time points (default); and the first, second, and last time point (specified through `index` argument):


```python
meta.pv_loop_growth(beats)
meta.pv_loop_growth(beats, index=(0, 1, -1))
```


    
![png](growth_files/growth_11_0.png)
    



    
![png](growth_files/growth_11_1.png)
    


So why did the LV grow? Remember that we are using stretch-driven growth relationships, so let's visualize the changes in maximum fiber stretch and stress in all three walls:


```python
meta.plot_growth(beats, ["MaxStretchLfw", "MaxStretchSw", "MaxStretchRfw", "MaxStressLfw", "MaxStressSw", "MaxStressRfw"], units=["-", "-", "-"])
```


    
![png](growth_files/growth_13_0.png)
    


From these figures, we anticipate that the left free wall (LFW) and septal wall (SW) will dilate due to the increase in maximum fiber stretch, and initially thin because of a decrease in maximum fiber stress. The right free wall (RFW) experience only very small changes in maximum fiber stretch and stress, and thus should not grow.

We can test our hypothesis by looking at the growth tensor *F_g*, which can be visualized using the `plot_fg` function. Indeed, note how there is only noticeable growth in the left free wall (LFW) and septal wall (SW) as mitral regurgitation only effects the mechanics in the left ventricle. As expected from experimental and clinical studies, we observe dilation (increase in fiber (11) and cross-fiber (22) components of the growth tensor) and wall thinning (decrease in the radial (33) component of the growth tensor) in the LFW and SW. Initially, the change in wall volume due to growth (*J_g*) decreases due to faster thinning than dilation, but eventually the wall volume increases due to dilation exceeding wall thinning.


```python
meta.plot_fg(beats)
```


    
![png](growth_files/growth_15_0.png)
    


### Isotropic growth
Let's now simulate isotropic growth in a canine model with mitral valve regurgitation. We will use the `isotropic_jones` growth model to simulate isotropic growth.


```python
# Initialize model and change growth type
beats = Hatch(input_canine)
beats.growth.type = "isotropic_jones"

# Simulate growth
beats.let_it_grow()
```

Use the standard plotting functions to visualize the results, note now how there is a smaller increase in LVEDV and even a decrease in LVESV compared to the transversely isotropic growth model.


```python
meta.plot_growth(beats, ["LVEDV", "LVESV"], units=["mL", "mL"])
meta.pv_loop_growth(beats)
```


    
![png](growth_files/growth_19_0.png)
    



    
![png](growth_files/growth_19_1.png)
    


This is due to the isotropic growth model not allowing for wall thinning:


```python
meta.plot_fg(beats)
```


    
![png](growth_files/growth_21_0.png)
    


## References
[1] Jones, C. E. & Oomen, P. J. A. Synergistic Biophysics and Machine Learning Modeling to Rapidly Predict Cardiac Growth Probability. bioRxiv 2024.07.17.603959 (2024) doi:10.1101/2024.07.17.603959.
[2] Oomen, P. J. A., Phung, T.-K. N., Weinberg, S. H., Bilchick, K. C. & Holmes, J. W. A rapid electromechanical model to predict reverse remodeling following cardiac resynchronization therapy. Biomech Model Mechan 21, 231–247 (2022).
[3] Witzenburg, C. M. & Holmes, J. W. Predicting the Time Course of Ventricular Dilation and Thickening Using a Rapid Compartmental Model. J Cardiovasc Transl 11, 109–122 (2018).
[4] Göktepe, S., Abilez, O. J., Parker, K. K. & Kuhl, E. A multiscale model for eccentric and concentric cardiac growth through sarcomerogenesis. J Theor Biol 265, 433–442 (2010).
[5] Rodriguez, E. K., Hoger, A. & McCulloch, A. D. Stress-dependent finite growth in soft elastic tissues. J Biomech 27, 455–467 (1994).


```python

```
