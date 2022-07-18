args = {

    'output_dir': 'temp_olid/outputs/',
    "best_model_dir": "temp_olid/outputs/best_model",
    'cache_dir': 'temp_olid/cache_dir/',

    "early_stopping": True,
    "early_stopping_min_delta" : 0.0001,
    "early_stopping_patience": 10,

    'embed_size': 300,

    'learning_rate': 1e-3,
    'max_features': None,
    'max_len': 256,

    'manual_seed': 777,

    'n_fold': 3,

    'num_classes': 2,
    'num_train_epochs': 50,

    'reduce_lr_on_plateau': True,
    'reduce_lr_on_plateau_factor': 0.6,
    'reduce_lr_on_plateau_patience':  2,
    'reduce_lr_on_plateau_min_lr':  0.0001,

    'save_best_model': True,

    'test_batch_size': 128,
    'train_batch_size': 128,

}