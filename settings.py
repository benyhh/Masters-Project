
patches = {
    0: 	None,
    1:	(-75,75,75,15),
    2:	(-28,28,76,60),
    3:	(-120,-98,63,16),
    4:	(-75,0,75,15),
    5: 	(-28,28,57,47),
    5: (-0.083, 0.057),
    6: (0.195, 0.266),
    7:  (0,90,90,0),
    8:  (90,180,90,0),
    9:  (-180,-90,90,0),
    10: (-90,0,90,0),
    11: (-180,180,75,15),
    12: (-180,180,75,45),
    13: (-180,180,45,15),

    } #


features = {
    'None':   None,

    'all':    ['ACTUALAZ', 'ACTUALEL', 'TEMP1', 'TEMP26', 'TEMP28', 'TILT1T', 'Az_sun', 'El_sun', 'SunElDiff',
            'SunAzDiff', 'SunAngleDiff', 'SunAngleDiff_15', 'POSITIONX', 'POSITIONY', 'PRESSURE', 'HUMIDITY',
            'WINDDIR DIFF', 'TURBULENCE', 'Hour', 'date'],

    'offset': ['TEMP1', 'TEMP26', 'TEMP28', 'TILT1T', 'Az_sun','SunAzDiff', 'POSITIONY', 'PRESSURE', 'Hour', 'date'],

    'sf2':    ['ACTUALAZ', 'ACTUALEL', 'TEMP1', 'TILT1T', 'SunAngleDiff', 'SunAngleDiff_15', 'POSITIONY', 'HUMIDITY',
            'TURBULENCE', 'Hour', 'date'],


    'Corr':   ['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
            'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
            'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
            'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','ROTATIONX_sumdabs2','ACTUALAZ_sumdabs2',
            'TILT1X_sumdabs2','ACTUALEL_sumdabs5','POSITIONX_sumdabs5','ROTATIONX_sumdabs5',
            'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'DAZ_TOTAL', 'DEL_TOTAL', 'date'],

    'Corr_reduced':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TEMP27','TILT1X','WINDDIRECTION',
                    'Az_sun','El_sun','SunAboveHorizon','SunAngleDiff','SunAngleDiff_15','SunElDiff',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1',
                    'DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],

    'Corr_reduced2':['ACTUALAZ','ACTUALEL','HUMIDITY','POSITIONZ','TEMP1','TILT1X','WINDDIRECTION',
                    'SunAngleDiff','SunAngleDiff_15','SunElDiff','Hour',
                    'TURBULENCE','WINDDIR DIFF','ACTUALEL_sumdabs1','TILT1X_sumdabs1','POSITIONX_sumdabs1',
                    'POSITIONZ_sumdabs1','ROTATIONX_sumdabs1','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT','date'],

    'Corr_reduced3':['ACTUALAZ','ACTUALEL','TEMP1','TILT1X','WINDDIRECTION','SunAngleDiff_15','Hour',
                    'TURBULENCE','WINDDIR DIFF','DAZ_TILT', 'DAZ_TILTTEMP', 'DAZ_DISP', 'DEL_DISP', 'DEL_TILT', 'date'],

    'hp_el1':          ['DEL_TILT', 'WINDDIRECTION', 'POSITIONZ', 'ACTUALAZ', 'HUMIDITY', 'SunAngleDiff_15', 'TILT1X_sumdabs1', 'WINDDIR DIFF', 'TURBULENCE', 'Hour', 'date'],            

    'hp_az0':          ['SunElDiff', 'DEL_TILT', 'DAZ_TILT', 'DAZ_TILTTEMP', 'TILT1X_sumdabs1', 'ACTUALEL', 'WINDDIRECTION',
                    'DAZ_DISP', 'SunElDiff', 'ACTUALEL', 'TILT1X', 'TURBULENCE', 'ROTATIONX_sumdabs1', 'TEMP1', 'WINDDIR DIFF','date'],
    'new': ['DISP_ABS2_MEDIAN','DISP_ABS3_MEDIAN', 'TEMP27_MEDIAN', 'TEMP28_MEDIAN', 'TEMP2_MEDIAN', 'TEMP3_MEDIAN', 'TEMP4_MEDIAN', 'TEMP5_MEDIAN', 'TEMP6_MEDIAN',
            'TILT2X_MEDIAN', 'TILT2Y_MEDIAN', 'TILT2T_MEDIAN', 'TILT3X_MEDIAN', 'TILT3Y_MEDIAN', 'TILT3T_MEDIAN',
            'WINDSPEED_VARIANCE_5', 'TEMP1_MAX_CHANGE_P5', 'TEMP26_MAX_CHANGE_P5', 'COMMANDAZ_MEDAN','COMMANDEL_MEDIAN',
            'TEMPERATURE_MEDIAN', 'HUMIDITY_MEDIAN', ],
    'az': ['date', 'ACTUALAZ_MEDIAN', 'ACTUALEL_MEDIAN', 'TEMP1_CHANGE', 'TEMP1', 'SUNAZ_MEDIAN', 'SUNEL_MEDIAN',
            'ACTUALAZ_POS_CHANGE', 'ACTUALAZ_NEG_CHANGE', 'TILT1X_MEDIAN', 'TILT1Y_MEDIAN', 'POSITIONZ_MEDIAN',
            'ACTUALEL_POS_CHANGE','ACTUALEL_NEG_CHANGE','DAZ_TILT_MEDIAN', 'DAZ_TEMP_MEDIAN', 'DAZ_DISP_MEDIAN',
            'WINDDIRECTION_MEDIAN', 'WINDSPEED_MEDIAN', 'HUMIDITY_MEDIAN'],
    'az2': ['date', 'ACTUALAZ_MEDIAN', 'ACTUALEL_MEDIAN', 'TEMP1_CHANGE', 'TEMP1', 'SUNAZ_MEDIAN', 'SUNEL_MEDIAN',
            'ACTUALAZ_POS_CHANGE', 'ACTUALAZ_NEG_CHANGE', 'TILT1X_MEDIAN', 'POSITIONZ_MEDIAN'],

    'el': ['date', 'ACTUALEL_MEDIAN', 'ACTUALAZ_MEDIAN', 'POSITIONZ_MEDIAN', 'ACTUALEL_POS_CHANGE', 'ACTUALEL_NEG_CHANGE','TEMP1_CHANGE',
            'TEMP1', 'SUNEL_MEDIAN', 'SUNAZ_MEDIAN', 'DEL_TILT_MEDIAN', 'DEL_TEMP_MEDIAN', 'HUMIDITY_MEDIAN', 'DISP_ABS1_MEDIAN'
            ],

    'optical1': ['TEMP1_MEDIAN_1', 'TILT1X_MEDIAN_1', 'TILT1Y_MEDIAN_1'],
    'optical2': ['WINDDIRECTION_MEDIAN_1', 'TILT1T_MEDIAN_1', 'POSITIONZ_MEDIAN_1',
                'POSITIONX_MEDIAN_1', 'POSITIONY_MEDIAN_1','ROTATIONX_MEDIAN_1', 'HUMIDITY_MEDIAN_1',
                'DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'DISP_ABS3_MEDIAN_1',
                ],
    'optical3': ['DISP_ABS1_MEDIAN_1', 'DISP_ABS2_MEDIAN_1', 'DISP_ABS3_MEDIAN_1'
                ],

    'new_model':['MODELAZ','MODELEL','COMMANDAZ_MEDIAN', 'COMMANDEL_MEDIAN', 'TEMP3_MEDIAN'],
    'optical': ['COMMANDAZ', 'COMMANDEL']

    }

dataset_params = {
    'use_pca'           : False,
    'pca_components'    : 0.998,

    'use_scaler'        : False,
    'scaler'            : 'StandardScaler',

    'use_cartesian'     : False,

    'use_patches'       : False,
    'patch_key'         : 1,

    'use_features'      : True,
    'feature_key'       : 'optical',

    'remove_outliers'   : False,
    'outlier_threshold' : 2.7,

    'filter_instruments' : False,
    'rx': 'NFLASH230',

    'target'            : 'real_az',

    'new_model' : False,
    'optical_model' : True


}