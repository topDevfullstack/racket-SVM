
#lang racket

; Racket implementation of "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM"
; http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf

; This is placed in the public domain, all rights are relinquished. Have at it.
; - Jay Kominek, 2011-10-10

(require racket/flonum)
(require (planet neil/levenshtein:1:3/levenshtein))

; performs one iteration/step of the pegasos algorithm
; this picks a random vector to optimize. if you wanted to rework
; this to work for massive data sets, this should be changed to
; take a single vector at a time, and the function invoking it
; should just feed them in efficiently
(define (pegasos-compute kernel vectors labels alphas Î» t)
  (let* ([i (random (vector-length vectors))]
         [yi (vector-ref labels i)]
         [xi (vector-ref vectors i)])
    (if
     (fl> 1.0
        (fl* yi (fl/ (fl* Î» t))
           (for/fold ([sum 0.0])
             ([alpha (in-vector alphas)]
              [y (in-vector labels)]
              [xj (in-vector vectors)]
              [j (in-naturals)]
              ;#:when (not (= i j))
              )
             (fl+ sum (* alpha y (kernel xi xj))))))
     i
     #f)))

; loops over the data set until done? indicates it's time to stop
(define (pegasos-optimize kernel vectors labels Î» done?)
  (let/ec exit
    (let ([alphas (make-vector (vector-length vectors))])
      (for ([t (in-naturals 1)])
        (let ([incr (pegasos-compute kernel vectors labels alphas Î» t)])
          (when incr
            (vector-set! alphas incr (add1 (vector-ref alphas incr))))
          (when (done? t alphas)
            (exit alphas)))))))

; evaluates an svm to get a decision
(define (evaluate v kernel vectors labels alphas)
  (for/fold ([sum 0.0])
    ([alpha (in-vector alphas)]
     [x (in-vector vectors)]
     [y (in-vector labels)])
    (fl+ sum (* alpha y (kernel x v)))))

; an implementation for done?
(define (iteration-limit n)
  (lambda (iter alphas)
    (> iter n)))

; an implementation for done?
(define (time-limit ms)
  (let ([start #f])
    (lambda (iter alphas)
      (if (not start)
          (begin
            (set! start (current-milliseconds))
            #f)
          (> (- (current-milliseconds) start) ms)))))

; Kernels are all any/c -> flonum?

(define (linear/dict a b)
  (for/fold ([sum 0.0])
    ([k (in-dict-keys a)])
    (if (dict-has-key? b k)
        (+ sum (* (dict-ref a k) (dict-ref b k)))
        sum)))

(define (linear/vector a b)
  (for/fold ([sum 0.0])
    ([av (in-vector a)]
     [bv (in-vector b)])
    (+ sum (* av bv))))

(define (linear/fl-dict a b)
  (for/fold ([sum 0.0])
    ([k (in-dict-keys a)])
    (if (dict-has-key? b k)
        (fl+ sum (fl* (dict-ref a k) (dict-ref b k)))
        sum)))

(define (linear/fl-vector a b)
  (for/fold ([sum 0.0])
    ([av (in-vector a)]
     [bv (in-vector b)])
    (fl+ sum (fl* av bv))))

(define (gaussian-rbf/vector gamma)
  (let ([ngamma (exact->inexact (- gamma))])
    (lambda (a b)
      (flexp (fl* ngamma
                  (for/fold ([sum 0.0])
                     ([av (in-vector a)]
                      [bv (in-vector b)])
                     (fl+ sum (expt (exact->inexact (- av bv)) 2)))))))

(define (edit-kernel levenshtein [gamma 0.5])
  (let ([g (exact->inexact gamma)])
    (lambda (a b)
      (flexp (fl- 0.0 (fl* g (exact->inexact (levenshtein a b))))))))
(define string-kernel (edit-kernel string-levenshtein 0.03125))


; Weee HOF's for making new kernels
(define (kernel+ k1 k2)
  (lambda (a b)
    (fl+ (k1 a b) (k2 a b))))

(define (kernel* k1 k2)
  (lambda (a b)
    (fl* (k1 a b) (k2 a b))))

; f : X -> Reals
(define (kernel-map f)
  (lambda (a b)
    (fl* (f a) (f b))))

(define (kernel-normalize k)
  (lambda (a b)
    (fl/ (k a b) (flsqrt (fl* (k a a) (k b b))))))




; two example data sets

(define data
  #( #(-2 -2)
     #(-2 -1)
     #(-1 -2)
     #(1 2)
     #(2 2)
     #(2 1) ))
(define labels
  #(-1 -1 1 -1 1 1))

(define string-data
  #( "add" "baa" "bad" "cab" "cad" "dab" "dad"     ; words matching /[abcd]{3}/
     "egg" "fee" "fie" "fig" "gee" "gig" "hie" ))  ; words matching /[efgh]{3}/
(define string-labels
  #( 1  1  1  1  1  1  1
    -1 -1 -1 -1 -1 -1 -1 ))

; the above training algorithm will produce rather large values in alphas,
; but they can be scaled down just fine           
(define (scale-alphas alphas)
  (let ([m (for/fold ([m 0])
             ([a (in-vector alphas)])
             (max m a))])
    (for/vector ([a (in-vector alphas)])
                (exact->inexact (/ a m)))))

(define alphas
  (scale-alphas
   (pegasos-optimize string-kernel string-data string-labels 0.001 (iteration-limit (/ 100 0.001)))))
(for ([v (in-vector string-data)]
      [truth (in-vector string-labels)])
  (printf "~a actual: ~a predicted: ~a~n" v truth (evaluate v string-kernel string-data string-labels alphas)))