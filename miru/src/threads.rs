//! A look's work spread over the threads it runs on: the device's, which `look` runs on
//! (`Device::install`). Each piece of work is one thread's alone and the results come in the
//! pieces' order, so what a look gives does not depend on how many threads there are.

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// `work` done for each of `0..count`, with the results in the order of their numbers, whichever
/// thread did which: on the threads the call runs on where it may use more than one (`threads`),
/// else on the calling thread alone.
pub(crate) fn each<T: Send>(
    threads: usize,
    count: usize,
    work: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    if threads.min(count) <= 1 {
        return (0..count).map(work).collect();
    }
    (0..count).into_par_iter().map(&work).collect()
}

/// `work` done for each of `pieces` with its number, on the threads the call runs on where it may
/// use more than one (`threads`), else on the calling thread alone: each piece is handed to one
/// thread, as the `&mut` pieces of one output can be, so that what is written is the same
/// whichever thread did which.
pub(crate) fn each_piece<P: Send>(threads: usize, pieces: Vec<P>, work: impl Fn(usize, P) + Sync) {
    if threads.min(pieces.len()) <= 1 {
        for (number, piece) in pieces.into_iter().enumerate() {
            work(number, piece);
        }
        return;
    }
    let pieces = pieces.into_par_iter().enumerate();
    pieces.for_each(|(number, piece)| work(number, piece));
}

/// How many of `threads` threads item `number` of `count` may use for work of its own, where the
/// items are done on them at once ([`each`]): an even share, the first items taking one more
/// where the threads do not divide evenly, and one where there are more items than threads.
pub(crate) fn share(threads: usize, count: usize, number: usize) -> usize {
    if count >= threads {
        1
    } else {
        threads / count + usize::from(number < threads % count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The results come in the order of their numbers on one thread, on several, and on more
    /// threads than there is work.
    #[test]
    fn the_results_come_in_the_order_of_their_numbers() {
        for threads in [1, 3, 64] {
            let squares = each(threads, 17, |number| number * number);
            let expected: Vec<usize> = (0..17).map(|number| number * number).collect();
            assert_eq!(squares, expected, "on {threads} threads");
        }
        assert!(each(4, 0, |number| number).is_empty());
    }

    /// Each piece is written by its own number on one thread, on several, and on more threads
    /// than there are pieces.
    #[test]
    fn each_piece_is_written_by_its_number() {
        for threads in [1, 3, 64] {
            let mut values = vec![0; 17];
            each_piece(threads, values.chunks_mut(4).collect(), |number, piece| {
                for (k, value) in piece.iter_mut().enumerate() {
                    *value = 4 * number + k;
                }
            });
            assert_eq!(values, (0..17).collect::<Vec<_>>(), "on {threads} threads");
        }
    }

    /// Two items on fourteen threads take seven each; three on eight take three, three and two;
    /// more items than threads take one each.
    #[test]
    fn the_threads_are_shared_out_evenly() {
        let shares = |threads, count| -> Vec<usize> {
            (0..count).map(|k| share(threads, count, k)).collect()
        };
        let expected: [(usize, usize, Vec<usize>); 4] = [
            (14, 2, vec![7, 7]),
            (8, 3, vec![3, 3, 2]),
            (2, 2, vec![1, 1]),
            (2, 5, vec![1; 5]),
        ];
        for (threads, count, expected) in expected {
            assert_eq!(shares(threads, count), expected, "{count} on {threads}");
        }
    }
}
