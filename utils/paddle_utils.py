import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
from typing import List
import yaml
from ast import literal_eval
import logging
import copy
from copy import deepcopy

def top_k_most_common(tensor, k=5):
    top_k_values = []
    for i in range(tensor.shape[0]):
        unique_values, counts = torch.unique(tensor[i], return_counts=True)
        sorted_indices = counts.argsort(descending=True)[:k]
        top_k_values.append(unique_values[sorted_indices])
    return torch.stack(top_k_values)

def most_common_value(tensor):
    values, counts = torch.unique(tensor, return_counts=True)
    max_count_index = counts.argmax()
    return values[max_count_index]

def get_one_hot(y_s):
    n_ways = torch.unique(y_s).size(0)

    eye = torch.eye(n_ways).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


def get_features(model, samples):
    model.eval()
    with torch.no_grad():
        features, _ = model(samples, True)
    #features = F.normalize(features.view(features.size(0), -1), dim=1)
    return features


def get_loss(logits_s, logits_q, labels_s, lambdaa):
    Q = logits_q.softmax(2)
    y_s_one_hot = get_one_hot(labels_s)
    ce_sup = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1)  # Taking the mean over samples within a task, and summing over all samples
    ent_q = get_entropy(Q)
    cond_ent_q = get_cond_entropy(Q)
    loss = - (ent_q - cond_ent_q) + lambdaa * ce_sup
    return loss


def get_mi(probs):
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)
    return q_cond_ent


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def wrap_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path),
                                )
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path),
                                )
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        #assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        if subkey in cfg:
            value = _decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, cfg[subkey], subkey, full_key
            )
            setattr(new_cfg, subkey, value)
        else:
            value = _decode_cfg_value(v)
            setattr(new_cfg, subkey, value)
    return new_cfg


class Logger:
    def __init__(self, module_name, filename):
        self.module_name = module_name
        self.filename = filename
        self.formatter = self.get_formatter()
        self.file_handler = self.get_file_handler()
        self.stream_handler = self.get_stream_handler()
        self.logger = self.get_logger()

    def get_formatter(self):
        log_format = '[%(name)s]: [%(levelname)s]: %(message)s'
        formatter = logging.Formatter(log_format)
        return formatter

    def get_file_handler(self):
        file_handler = logging.FileHandler(self.filename)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        return stream_handler

    def get_logger(self):
        logger = logging.getLogger(self.module_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.file_handler)
        logger.addHandler(self.stream_handler)
        return logger

    def del_logger(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def exception(self, msg):
        self.logger.exception(msg)


def make_log_dir(log_path, dataset, backbone, method):
    log_dir = os.path.join(log_path, dataset, backbone, method)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_file(log_path, dataset, backbone, method):
    log_dir = make_log_dir(log_path=log_path, dataset=dataset, backbone=backbone, method=method)
    i = 0
    filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
    while os.path.exists(os.path.join(log_dir, '{}_run_%s.log'.format(method)) % i):
        i += 1
        filename = os.path.join(log_dir, '{}_run_{}.log'.format(method, i))
    return filename


def extract_features(model, model_path, model_tag, used_set, used_set_name, fresh_start, loaders_dic):
    """
    inputs:
        model : The loaded model containing the feature extractor
        loaders_dic : Dictionnary containing training and testing loaders
        model_path : Where was the model loaded from
        model_tag : Which model ('final' or 'best') to load
        used_set : Set used between 'test' and 'val'
        k_eff : Number of ways for the task

    returns :
        extracted_features_dic : Dictionnary containing all extracted features and labels
    """

    # Load features from memory if previously saved ...
    save_dir = os.path.join(model_path, model_tag, used_set_name)
    filepath = os.path.join(save_dir, 'output.plk')
    if os.path.isfile(filepath) and (not fresh_start):
        extracted_features_dic = load_pickle(filepath)
        print(" ==> Features loaded from {}".format(filepath))
        return extracted_features_dic

    # ... otherwise just extract them
    else:
        print(" ==> Beginning feature extraction")
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():

        all_features = []
        all_labels = []
        tqdm_loop = tqdm(loaders_dic[used_set])
        for i, (inputs, labels, _) in enumerate(tqdm_loop):
            inputs = inputs.to(self.device).unsqueeze(0)
            labels = torch.Tensor([labels])
            outputs, _ = model(inputs, True)
            all_features.append(outputs.cpu())
            all_labels.append(labels)
        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)
        extracted_features_dic = {'concat_features': all_features,
                                    'concat_labels': all_labels
                                    }
    print(" ==> Saving features to {}".format(filepath))
    save_pickle(filepath, extracted_features_dic)
    return extracted_features_dic


def extract_mean_features(model, train_loader, args, logger, device):
    """
        inputs:
            model : The loaded model containing the feature extractor
            train_loader : Train data loader
            args : arguments
            logger : logger object
            device : GPU device

        returns :
            out_mean : Training data features mean
    """

    # Load features from memory if previously saved ...
    save_dir = os.path.join(args.ckpt_path, args.model_tag, args.used_set_support)
    filepath_mean = os.path.join(save_dir, 'output_mean.plk')

    # get training mean
    if not os.path.isfile(filepath_mean):

        logger.info(" ==> Beginning feature extraction to compute training mean")
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            out_mean, fc_out_mean = [], []
            for i, (inputs, labels, _) in enumerate(wrap_tqdm(train_loader, False)):
                inputs = inputs.to(device)
                outputs, fc_outputs = model(inputs, True)
                out_mean.append(outputs.cpu().data.numpy())
                if fc_outputs is not None:
                    fc_out_mean.append(fc_outputs.cpu().data.numpy())
            out_mean = np.concatenate(out_mean, axis=0).mean(0)
            if len(fc_out_mean) > 0:
                fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
            else:
                fc_out_mean = -1
            logger.info(" ==> Saving features to {}".format(filepath_mean))
            save_pickle(save_dir + '/output_mean.plk', [out_mean, fc_out_mean])
            return  torch.from_numpy(out_mean),  torch.from_numpy(fc_out_mean)
    else:
        out_mean, fc_out_mean = load_pickle(save_dir + '/output_mean.plk')
        logger.info(" ==> Features loaded from {}".format(filepath_mean))
        return torch.from_numpy(out_mean), torch.from_numpy(fc_out_mean)
    
# Inverse Covariance Estimator : Graphical LASSO
def prox_g(S, gamma):                                       # g(S) = -logdet(S)   
    s, U = torch.linalg.eigh(S)                             # eighenvalues <-> determinant de matrice diagonalisable 
    d = (s + torch.sqrt(s**2 + 4 * gamma)) / 2
    P = (U * d.unsqueeze(-2)).matmul(U.transpose(-1, -2))
    return P

def prox_f(S, gamma):                                        # f(S) = lambda*sum_{i!=j}|Sij|
    m = torch.diag(S.new_ones(S.size(-1), dtype=bool))       # True sur diagonale, False ailleurs
    m_bar = ~m                                               # False sur diagonale, True ailleurs
    S[..., m_bar] = F.softshrink(S[..., m_bar], lambd=gamma) # softshrink(x)=x-lambda if x-lambda>0, x+lambda if x+lambda>0, and 0 otherwise
    S[..., m] = F.softshrink(S[..., m], lambd=0)             # useless i guess ?
    return S

def GLASSO(C, S_0, lambd, max_iter=20000, eps=5e-7):
    """
    inputs:
        C : torch.Tensor of shape [feature_dim, feature_dim]
        S_0 : torch.Tensor of shape [feature_dim, feature_dim]
        lambd : scalar
    """    
    y_n = x_n = S_0.clone()
    n_task, feature_dim = S_0.size(0), S_0.size(-1)
    gamma = 1
    lambda_n = 1
    criterion = 1
    i = 0
    for i in range(max_iter):
        x_n_plus_1 = prox_g(y_n - gamma * C, gamma)          # prox_gamma * g(y_n - gamma*C) = prox_{ gamma*g(.) + <C|.> } (y_n)   
        #criterion = torch.norm(x_n_plus_1 - x_n)**2 / (feature_dim**2)
        #if i % 100 == 0 and i > 0:
        #    print("iter ", i, "criterion = ", criterion)
        # if criterion < eps:
        #     break
        x_n = deepcopy(x_n_plus_1)
        Z = prox_f(2 * x_n - y_n,  gamma * lambd)    
        y_n = y_n + lambda_n * (Z - x_n)
        criterion = torch.norm(Z - x_n)**2 / (feature_dim**2)
        #print('criterion', criterion)
        if criterion < eps:
            break
        
    #if i >= max_iter:
    #    print("GLASSO stopped after max number of iterations")
    #else:
    #    print("GLASSO completed in ", i, "iterations")
    #return x_n
    return Z