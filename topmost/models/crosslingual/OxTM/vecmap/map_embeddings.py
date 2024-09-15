# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import collections
import re
import sys
import time
from types import SimpleNamespace

import numpy as np


def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def write(words, matrix, file):
    m = np.asarray(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = np
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = np
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    xp = np
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = np
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = np
        mask = xp.random.rand(*m.shape) >= p
        return m * mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = np
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def map_supervised_embeddings(
        train_dict,
        src_emb,
        trg_emb,
        src_mapped_emb,
        trg_mapped_emb
):
    # Parse command line arguments
    args = SimpleNamespace(
        src_input=src_emb,  # 기본값으로 src_emb 할당
        trg_input=trg_emb,  # 기본값으로 trg_emb 할당
        src_output=src_mapped_emb,  # 기본값으로 src_mapped_emb 할당
        trg_output=trg_mapped_emb,  # 기본값으로 trg_mapped_emb 할당
        encoding='utf-8',
        precision='fp32',
        cuda=False,
        batch_size=1000,
        seed=0,
        supervised=train_dict,
        semi_supervised=None,
        identical=False,
        unsupervised=False,
        acl2018=False,
        aaai2018=None,
        acl2017=False,
        acl2017_seed=None,
        emnlp2016=None,
        init_dictionary=train_dict,
        init_identical=False,
        init_numerals=False,
        init_unsupervised=False,
        unsupervised_vocab=0,
        normalize=['unit', 'center', 'unit'],
        whiten=True,
        src_reweight=0.5,
        trg_reweight=0.5,
        src_dewhiten='src',
        trg_dewhiten='trg',
        dim_reduction=0,
        orthogonal=False,
        unconstrained=False,
        self_learning=False,
        vocabulary_cutoff=0,
        direction='union',
        csls_neighborhood=0,
        threshold=0.000001,
        validation=None,
        stochastic_initial=0.1,
        stochastic_multiplier=2.0,
        stochastic_interval=50,
        log=None,
        verbose=False
    )

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = read(srcfile, dtype=dtype)
    trg_words, z = read(trgfile, dtype=dtype)

    xp = np
    xp.random.seed(args.seed)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    normalize(x, args.normalize)
    normalize(z, args.normalize)

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0],
                                                                                        args.unsupervised_vocab)
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u * s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u * s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        normalize(xsim, args.normalize)
        normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)
        if args.csls_neighborhood > 0:
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            sim -= knn_sim_fwd[:, xp.newaxis] / 2 + knn_sim_bwd / 2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
    elif args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # Read validation dictionary
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    # Allocate memory
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    simfwd = xp.empty((args.batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((args.batch_size, src_size), dtype=dtype)
    if args.validation is not None:
        simval = xp.empty((len(validation.keys()), z.shape[0]), dtype=dtype)

    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = not args.self_learning
    while True:

        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier * keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if args.orthogonal or not end:  # orthogonal mapping
            u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        elif args.unconstrained:  # unconstrained mapping
            x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping

            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1 / s)).dot(vt)

            if args.whiten:
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s ** args.src_reweight
            zw *= s ** args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if args.trg_dewhiten == 'src':
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if args.dim_reduction > 0:
                xw = xw[:, :args.dim_reduction]
                zw = zw[:, :args.dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                    simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j - i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                    simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j - i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            # Accuracy and similarity evaluation in validation
            if args.validation is not None:
                src = list(validation.keys())
                xw[src].dot(zw.T, out=simval)
                nn = np.asarray(simval.argmax(axis=1))
                accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
                similarity = np.mean(
                    [max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob), file=sys.stderr)
                if args.validation is not None:
                    print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                    print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                    print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity, 100 * accuracy, 100 * validation_coverage) if args.validation is not None else ''
                print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)
                log.flush()

        t = time.time()
        it += 1

    # Write mapped embeddings
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    write(src_words, xw, srcfile)
    write(trg_words, zw, trgfile)
    srcfile.close()
    trgfile.close()


if __name__ == '__main__':
    map_supervised_embeddings(
        train_dict='./data/dictionaries/en-de.train.txt',
        src_emb='./data/embeddings/en.emb.txt',
        trg_emb='./data/embeddings/de.emb.txt',
        src_mapped_emb='./en.emb',
        trg_mapped_emb='./de.emb',
    )
