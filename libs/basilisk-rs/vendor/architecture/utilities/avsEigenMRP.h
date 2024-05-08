/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

/// \cond DO_NOT_DOCUMENT


#ifndef EIGEN_MRP_H
#define EIGEN_MRP_H

//#include <Eigen/Core>
//#include <Eigen/src/Core//util//DisableStupidWarnings.h>
//#include <Eigen/SVD>
//#include <Eigen/LU>
//#include "Eigen/src/Geometry/Quaternion.h"

namespace Eigen {
    template<typename Scalar, int Options = AutoAlign> class MRP;


    
    /***************************************************************************
     * Definition of MRPBase<Derived>
     * The implementation is at the end of the file
     ***************************************************************************/

    namespace internal {
        template<typename Other,
        int OtherRows=Other::RowsAtCompileTime,
        int OtherCols=Other::ColsAtCompileTime>
        struct MRPbase_assign_impl;
    }

    /** \geometry_module \ingroup Geometry_Module
     * \class MRPBase
     * \brief Base class for MRP expressions
     * \tparam Derived derived type (CRTP)
     * \sa class MRP
     */
    template<class Derived>
    class MRPBase : public RotationBase<Derived, 3>
    {
        typedef RotationBase<Derived, 3> Base;
    public:
        using Base::operator*;
        using Base::derived;

        typedef typename internal::traits<Derived>::Scalar Scalar;              //!< variable
        typedef typename NumTraits<Scalar>::Real RealScalar;                    //!< variable
        typedef typename internal::traits<Derived>::Coefficients Coefficients;  //!< variable
        enum {
            Flags = Eigen::internal::traits<Derived>::Flags
        };

        // typedef typename Matrix<Scalar,3,1> Coefficients;
        /** the type of a 3D vector */
        typedef Matrix<Scalar,3,1> Vector3;
        /** the equivalent rotation matrix type */
        typedef Matrix<Scalar,3,3> Matrix3;
        /** the equivalent angle-axis type */
        typedef AngleAxis<Scalar> AngleAxisType;

        /** default constructor */
        MRPBase() = default;
        /** copy constructor */
        MRPBase(const MRPBase<Derived> &/*rhs*/) = default;

        /** \returns the \c x coefficient */
        inline Scalar x() const { return this->derived().coeffs().coeff(0); }
        /** \returns the \c y coefficient */
        inline Scalar y() const { return this->derived().coeffs().coeff(1); }
        /** \returns the \c z coefficient */
        inline Scalar z() const { return this->derived().coeffs().coeff(2); }

        /** \returns a reference to the \c x coefficient */
        inline Scalar& x() { return this->derived().coeffs().coeffRef(0); }
        /** \returns a reference to the \c y coefficient */
        inline Scalar& y() { return this->derived().coeffs().coeffRef(1); }
        /** \returns a reference to the \c z coefficient */
        inline Scalar& z() { return this->derived().coeffs().coeffRef(2); }

        /** \returns a read-only vector expression of the imaginary part (x,y,z) */
        inline const VectorBlock<const Coefficients,3> vec() const { return coeffs().template head<3>(); }

        /** \returns a vector expression of the imaginary part (x,y,z) */
        inline VectorBlock<Coefficients,3> vec() { return coeffs().template head<3>(); }

        /** \returns a read-only vector expression of the coefficients (x,y,z) */
        inline const typename internal::traits<Derived>::Coefficients& coeffs() const { return derived().coeffs(); }

        /** \returns a vector expression of the coefficients (x,y,z) */
        inline typename internal::traits<Derived>::Coefficients& coeffs() { return derived().coeffs(); }

        EIGEN_STRONG_INLINE MRPBase<Derived>& operator=(const MRPBase<Derived>& other); //!< method
        template<class OtherDerived> EIGEN_STRONG_INLINE Derived& operator=(const MRPBase<OtherDerived>& other); //!< method

        // disabled this copy operator as it is giving very strange compilation errors when compiling
        // test_stdvector with GCC 4.4.2. This looks like a GCC bug though, so feel free to re-enable it if it's
        // useful; however notice that we already have the templated operator= above and e.g. in MatrixBase
        // we didn't have to add, in addition to templated operator=, such a non-templated copy operator.
        //  Derived& operator=(const MRPBase& other)
        //  { return operator=<Derived>(other); }

        Derived& operator=(const AngleAxisType& aa);
        template<class OtherDerived> Derived& operator=(const MatrixBase<OtherDerived>& m); //!< method

        /** \returns a MRP representing an identity rotation
         * \sa MatrixBase::Identity()
         */
        static inline MRP<Scalar> Identity() { return MRP<Scalar>(Scalar(1), Scalar(0), Scalar(0), Scalar(0)); }

        /** \sa MRPBase::Identity(), MatrixBase::setIdentity()
         */
        inline MRPBase& setIdentity() { coeffs() << Scalar(0), Scalar(0), Scalar(0); return *this; }

        /** \returns the squared norm of the MRP's coefficients
         * \sa MRPBase::norm(), MatrixBase::squaredNorm()
         */
        inline Scalar squaredNorm() const { return coeffs().squaredNorm(); }

        /** \returns the norm of the MRP's coefficients
         * \sa MRPBase::squaredNorm(), MatrixBase::norm()
         */
        inline Scalar norm() const { return coeffs().norm(); }

        /** Normalizes the MRP \c *this
         * \sa normalized(), MatrixBase::normalize() */
        inline void normalize() { coeffs().normalize(); }
        /** \returns a normalized copy of \c *this
         * \sa normalize(), MatrixBase::normalized() */
        inline MRP<Scalar> normalized() const { return MRP<Scalar>(coeffs().normalized()); }

        /** \returns the dot product of \c *this and \a other
         * Geometrically speaking, the dot product of two unit MRPs
         * corresponds to the cosine of half the angle between the two rotations.
         * \sa angularDistance()
         */
        template<class OtherDerived> inline Scalar dot(const MRPBase<OtherDerived>& other) const { return coeffs().dot(other.coeffs()); }

        template<class OtherDerived> Scalar angularDistance(const MRPBase<OtherDerived>& other) const; //!< method

        /** \returns an equivalent 3x3 rotation matrix */
        Matrix3 toRotationMatrix() const;

        /** \returns the MRP which transform \a a into \a b through a rotation */
        template<typename Derived1, typename Derived2>
        Derived& setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b);

        template<class OtherDerived> EIGEN_STRONG_INLINE MRP<Scalar> operator* (const MRPBase<OtherDerived>& q) const; //!< method
        template<class OtherDerived> EIGEN_STRONG_INLINE Derived& operator*= (const MRPBase<OtherDerived>& q);
        template<class OtherDerived> EIGEN_STRONG_INLINE Derived& operator+= (const MRPBase<OtherDerived>& q);
        template<class OtherDerived> EIGEN_STRONG_INLINE Derived& operator-= (const MRPBase<OtherDerived>& q);

        /** \returns the MRP describing the shadow set */
        MRP<Scalar> shadow() const;

        /** \returns 3x3 [B] matrix for the MRP differential kinematic equations */
        Matrix3 Bmat() const;

        /** \returns the MRP describing the inverse rotation */
        MRP<Scalar> inverse() const;

        /** \returns the conjugated MRP */
        MRP<Scalar> conjugate() const;

        //        template<class OtherDerived> MRP<Scalar> slerp(const Scalar& t, const MRPBase<OtherDerived>& other) const;

        /** \returns \c true if \c *this is approximately equal to \a other, within the precision
         * determined by \a prec.
         *
         * \sa MatrixBase::isApprox() */
        template<class OtherDerived>
        bool isApprox(const MRPBase<OtherDerived>& other, const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const
        { return coeffs().isApprox(other.coeffs(), prec); }

        /** return the result vector of \a v through the rotation*/
        EIGEN_STRONG_INLINE Vector3 _transformVector(const Vector3& v) const;

        /** \returns \c *this with scalar type casted to \a NewScalarType
         *
         * Note that if \a NewScalarType is equal to the current scalar type of \c *this
         * then this function smartly returns a const reference to \c *this.
         */
        template<typename NewScalarType>
        inline typename internal::cast_return_type<Derived, MRP<NewScalarType> >::type cast() const
        {
            return typename internal::cast_return_type<Derived,MRP<NewScalarType> >::type(derived());
        }

    };

    /***************************************************************************
     * Definition/implementation of MRP<Scalar>
     ***************************************************************************/

    /** \geometry_module \ingroup Geometry_Module
     *
     * \class MRP
     *
     * \brief The MRP class used to represent 3D orientations and rotations
     *
     * \tparam _Scalar the scalar type, i.e., the type of the coefficients
     * \tparam _Options controls the memory alignment of the coefficients. Can be \# AutoAlign or \# DontAlign. Default is AutoAlign.
     *
     * This class represents a MRP \f$ (x,y,z) \f$ that is a convenient representation of
     * orientations and rotations of objects in three dimensions. Compared to other representations
     * like Euler angles or 3x3 matrices, MRPs offer the following advantages:
     * \li \b compact storage (3 scalars)
     * \li \b efficient to compose (28 flops),
     *
     * The following two typedefs are provided for convenience:
     * \li \c MRPf for \c float
     * \li \c MRPd for \c double
     *
     * \warning Operations interpreting the MRP as rotation have undefined behavior if the MRP is not normalized.
     *
     * \sa  class AngleAxis, class Transform
     */

    namespace internal {
        template<typename _Scalar,int _Options>
        /*! structure definition */
        struct traits<MRP<_Scalar,_Options> >
        {
            typedef MRP<_Scalar,_Options> PlainObject;
            typedef _Scalar Scalar;
            typedef Matrix<_Scalar,3,1,_Options> Coefficients;
            enum{
                IsAligned = internal::traits<Coefficients>::Flags & PacketAccessBit,
                Flags = IsAligned ? (PacketAccessBit | LvalueBit) : LvalueBit
            };
        };
    }

    template<typename _Scalar, int _Options>
    class MRP : public MRPBase<MRP<_Scalar,_Options> >
    {
        typedef MRPBase<MRP<_Scalar,_Options> > Base;
        enum { IsAligned = internal::traits<MRP>::IsAligned };

    public:
        typedef _Scalar Scalar; //!< variable

        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MRP)
        using Base::operator*=;

        typedef typename internal::traits<MRP>::Coefficients Coefficients;  //!< variable
        typedef typename Base::AngleAxisType AngleAxisType;                 //!< variable

        /** Default constructor leaving the MRP uninitialized. */
        inline MRP() {}

        /** Constructs and initializes the MRP \f$ (x,y,z) \f$ from
         * its four coefficients \a x, \a y and \a z.
         */
        inline MRP(const Scalar& x, const Scalar& y, const Scalar& z) : m_coeffs(x, y, z){}

        /** Constructs and initialize a MRP from the array data */
        inline MRP(const Scalar* data) : m_coeffs(data) {}

        /** Copy constructor */
        template<class Derived> EIGEN_STRONG_INLINE MRP(const MRPBase<Derived>& other) { this->Base::operator=(other); }

        /** Constructs and initializes a MRP from the angle-axis \a aa */
        explicit inline MRP(const AngleAxisType& aa) { *this = aa; }

        /** Constructs and initializes a MRP from either:
         *  - a rotation matrix expression,
         *  - a 3D vector expression representing MRP coefficients.
         */
        template<typename Derived>
        explicit inline MRP(const MatrixBase<Derived>& other) { *this = other; }

        /** Explicit copy constructor with scalar conversion */
        //        template<typename OtherScalar, int OtherOptions>
        //        explicit inline MRP(const MRP<OtherScalar, OtherOptions>& other)
        //        { m_coeffs = other.coeffs().template cast<Scalar>(); }

        template<typename Derived1, typename Derived2>
        static MRP FromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b);

        inline Coefficients& coeffs() { return m_coeffs;} //!< method
        inline const Coefficients& coeffs() const { return m_coeffs;} //!< method

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool(IsAligned))

    protected:
        Coefficients m_coeffs; //!< variable

#ifndef EIGEN_PARSED_BY_DOXYGEN
        static EIGEN_STRONG_INLINE void _check_template_params()         //!< method
        {
            EIGEN_STATIC_ASSERT( (_Options & DontAlign) == _Options,
                                INVALID_MATRIX_TEMPLATE_PARAMETERS)
        }
#endif
    };

    /** \ingroup Geometry_Module
     * single precision MRP type */
    typedef MRP<float> MRPf;
    /** \ingroup Geometry_Module
     * double precision MRP type */
    typedef MRP<double> MRPd;

    /***************************************************************************
     * Specialization of Map<MRP<Scalar>>
     ***************************************************************************/

    namespace internal {
        template<typename _Scalar, int _Options>
        /*! struct definition */
        struct traits<Map<MRP<_Scalar>, _Options> > : traits<MRP<_Scalar, (int(_Options)&Aligned)==Aligned ? AutoAlign : DontAlign> >
        {
            typedef Map<Matrix<_Scalar,3,1>, _Options> Coefficients;
        };
    }

    namespace internal {
        template<typename _Scalar, int _Options>
        /*! struct definition */
        struct traits<Map<const MRP<_Scalar>, _Options> > : traits<MRP<_Scalar, (int(_Options)&Aligned)==Aligned ? AutoAlign : DontAlign> >
        {
            typedef Map<const Matrix<_Scalar,3,1>, _Options> Coefficients;
            typedef traits<MRP<_Scalar, (int(_Options)&Aligned)==Aligned ? AutoAlign : DontAlign> > TraitsBase;
            enum {
                Flags = TraitsBase::Flags & ~LvalueBit
            };
        };
    }

    /** \ingroup Geometry_Module
     * \brief MRP expression mapping a constant memory buffer
     *
     * \tparam _Scalar the type of the MRP coefficients
     * \tparam _Options see class Map
     *
     * This is a specialization of class Map for MRP. This class allows to view
     * a 3 scalar memory buffer as an Eigen's MRP object.
     *
     * \sa class Map, class MRP, class MRPBase
     */
    template<typename _Scalar, int _Options>
    class Map<const MRP<_Scalar>, _Options >
    : public MRPBase<Map<const MRP<_Scalar>, _Options> >
    {
        typedef MRPBase<Map<const MRP<_Scalar>, _Options> > Base;

    public:
        typedef _Scalar Scalar;     //!< variable
        typedef typename internal::traits<Map>::Coefficients Coefficients; //!< variable
        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
        using Base::operator*=;

        /** Constructs a Mapped MRP object from the pointer \a coeffs
         *
         * The pointer \a coeffs must reference the three coefficients of MRP in the following order:
         * \code *coeffs == {x, y, z} \endcode
         *
         * If the template parameter _Options is set to #Aligned, then the pointer coeffs must be aligned. */
        EIGEN_STRONG_INLINE Map(const Scalar* coeffs) : m_coeffs(coeffs) {}

        inline const Coefficients& coeffs() const { return m_coeffs;} //!< method

    protected:
        const Coefficients m_coeffs; //!< variable
    };

    /** \ingroup Geometry_Module
     * \brief Expression of a MRP from a memory buffer
     *
     * \tparam _Scalar the type of the MRP coefficients
     * \tparam _Options see class Map
     *
     * This is a specialization of class Map for MRP. This class allows to view
     * a 3 scalar memory buffer as an Eigen's  MRP object.
     *
     * \sa class Map, class MRP, class MRPBase
     */
    template<typename _Scalar, int _Options>
    class Map<MRP<_Scalar>, _Options >
    : public MRPBase<Map<MRP<_Scalar>, _Options> >
    {
        typedef MRPBase<Map<MRP<_Scalar>, _Options> > Base;

    public:
        typedef _Scalar Scalar; //!< variable
        typedef typename internal::traits<Map>::Coefficients Coefficients; //!< variable
        EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
        using Base::operator*=;

        /** Constructs a Mapped MRP object from the pointer \a coeffs
         *
         * The pointer \a coeffs must reference the three coefficients of MRP in the following order:
         * \code *coeffs == {x, y, z} \endcode
         *
         * If the template parameter _Options is set to #Aligned, then the pointer coeffs must be aligned. */
        EIGEN_STRONG_INLINE Map(Scalar* coeffs) : m_coeffs(coeffs) {}

        inline Coefficients& coeffs() { return m_coeffs; } //!< method
        inline const Coefficients& coeffs() const { return m_coeffs; } //!< method

    protected:
        Coefficients m_coeffs; //!< variable
    };

    /** \ingroup Geometry_Module
     * Map an unaligned array of single precision scalars as a MRP */
    typedef Map<MRP<float>, 0>         MRPMapf;
    /** \ingroup Geometry_Module
     * Map an unaligned array of double precision scalars as a MRP */
    typedef Map<MRP<double>, 0>        MRPMapd;
    /** \ingroup Geometry_Module
     * Map a 16-byte aligned array of single precision scalars as a MRP */
    typedef Map<MRP<float>, Aligned>   MRPMapAlignedf;
    /** \ingroup Geometry_Module
     * Map a 16-byte aligned array of double precision scalars as a MRP */
    typedef Map<MRP<double>, Aligned>  MRPMapAlignedd;

    /***************************************************************************
     * Implementation of MRPBase methods
     ***************************************************************************/

    // Generic MRP * MRP product
    // This product can be specialized for a given architecture via the Arch template argument.
    namespace internal {
        /*! template definition */
        template<int Arch, class Derived1, class Derived2, typename Scalar, int _Options> struct mrp_product
        {
            static EIGEN_STRONG_INLINE MRP<Scalar> run(const MRPBase<Derived1>& a, const MRPBase<Derived2>& b){
                Scalar det;     //!< variable
                Scalar s1N2;    //!< variable
                Scalar s2N2;    //!< variable
                MRP<Scalar> s2 = b;
                MRP<Scalar> answer;
                s1N2 = a.squaredNorm();
                s2N2 = s2.squaredNorm();
                det = Scalar(1.0) + s1N2*s2N2 - 2*a.dot(b);
                if (det < 0.01) {
                    s2 = s2.shadow();
                    s2N2 = s2.squaredNorm();
                    det = Scalar(1.0) + s1N2*s2N2 - 2*a.dot(b);
                }
                answer = MRP<Scalar> (((1-s1N2)*s2.vec() + (1-s2N2)*a.vec() - 2*b.vec().cross(a.vec()))/det);
                if (answer.squaredNorm() > 1)
                    answer = answer.shadow();
                return answer;
            }
        };
    }

    /** \returns the concatenation of two rotations as a MRP-MRP product */
    template <class Derived>
    template <class OtherDerived>
    EIGEN_STRONG_INLINE MRP<typename internal::traits<Derived>::Scalar>
    MRPBase<Derived>::operator* (const MRPBase<OtherDerived>& other) const
    {
        EIGEN_STATIC_ASSERT((internal::is_same<typename Derived::Scalar, typename OtherDerived::Scalar>::value),
                            YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
        return internal::mrp_product<Architecture::Target, Derived, OtherDerived,
        typename internal::traits<Derived>::Scalar,
        internal::traits<Derived>::IsAligned && internal::traits<OtherDerived>::IsAligned>::run(*this, other);
    }

    /** \sa operator*(MRP) */
    template <class Derived>
    template <class OtherDerived>
    EIGEN_STRONG_INLINE Derived& MRPBase<Derived>::operator*= (const MRPBase<OtherDerived>& other)
    {
        derived() = derived() * other.derived();
        return derived();
    }

    /** \sa operator*(MRP) */
    template <class Derived>
    template <class OtherDerived>
    EIGEN_STRONG_INLINE Derived& MRPBase<Derived>::operator+= (const MRPBase<OtherDerived>& other)
    {
        derived().coeffs() = derived().coeffs() + other.derived().coeffs();
        return derived();
    }
    /** \sa operator*(MRP) */
    template <class Derived>
    template <class OtherDerived>
    EIGEN_STRONG_INLINE Derived& MRPBase<Derived>::operator-= (const MRPBase<OtherDerived>& other)
    {
        derived().coeffs() = derived().coeffs() - other.derived().coeffs();
        return derived();
    }


    /** Rotation of a vector by a MRP.
     * \remarks If the MRP is used to rotate several points (>1)
     * then it is much more efficient to first convert it to a 3x3 Matrix.
     * Comparison of the operation cost for n transformations:
     *   - MRP2:    30n
     *   - Via a Matrix3: 24 + 15n
     */
    template <class Derived>
    EIGEN_STRONG_INLINE typename MRPBase<Derived>::Vector3
    MRPBase<Derived>::_transformVector(const Vector3& v) const
    {
        // Note that this algorithm comes from the optimization by hand
        // of the conversion to a Matrix followed by a Matrix/Vector product.
        // It appears to be much faster than the common algorithm found
        // in the literature (30 versus 39 flops). It also requires two
        // Vector3 as temporaries.
        Vector3 uv = this->vec().cross(v);
        uv += uv;
        return v + this->w() * uv + this->vec().cross(uv);
    }

    template<class Derived>
    EIGEN_STRONG_INLINE MRPBase<Derived>& MRPBase<Derived>::operator=(const MRPBase<Derived>& other)
    {
        coeffs() = other.coeffs();
        return derived();
    }

    template<class Derived>
    template<class OtherDerived>
    EIGEN_STRONG_INLINE Derived& MRPBase<Derived>::operator=(const MRPBase<OtherDerived>& other)
    {
        coeffs() = other.coeffs();
        return derived();
    }

    /** Set \c *this from an angle-axis \a aa and returns a reference to \c *this
     */
    template<class Derived>
    EIGEN_STRONG_INLINE Derived& MRPBase<Derived>::operator=(const AngleAxisType& aa)
    {
        using std::tan;
        Scalar tanPhi = Scalar(0.25)*aa.angle(); // Scalar(0.25) to suppress precision loss warnings
        this->vec() = tanPhi * aa.axis();
        return derived();
    }

    /** Set \c *this from the expression \a xpr:
     *   - if \a xpr is a 3x1 vector, then \a xpr is assumed to be a MRP
     *   - if \a xpr is a 3x3 matrix, then \a xpr is assumed to be rotation matrix
     *     and \a xpr is converted to a MRP
     */

    template<class Derived>
    template<class MatrixDerived>
    inline Derived& MRPBase<Derived>::operator=(const MatrixBase<MatrixDerived>& xpr)
    {
        EIGEN_STATIC_ASSERT((internal::is_same<typename Derived::Scalar, typename MatrixDerived::Scalar>::value),
                            YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
        internal::MRPbase_assign_impl<MatrixDerived>::run(*this, xpr.derived());
        return derived();
    }

    /** Convert the MRP sigma_B/N to a 3x3 rotation matrix [NB].
     */
    template<class Derived>
    inline typename MRPBase<Derived>::Matrix3
    MRPBase<Derived>::toRotationMatrix(void) const
    {
        // NOTE if inlined, then gcc 4.2 and 4.4 get rid of the temporary (not gcc 4.3 !!)
        // if not inlined then the cost of the return by value is huge ~ +35%,
        // however, not inlining this function is an order of magnitude slower, so
        // it has to be inlined, and so the return by value is not an issue
        Matrix3 res;

        const Scalar ps2   = Scalar(1) + this->coeffs().squaredNorm();
        const Scalar ms2   = Scalar(1) - this->coeffs().squaredNorm();
        const Scalar ms2Sq = ms2 * ms2;
        const Scalar s1s2  = Scalar(8)*this->x()*this->y();
        const Scalar s1s3  = Scalar(8)*this->x()*this->z();
        const Scalar s2s3  = Scalar(8)*this->y()*this->z();
        const Scalar s1Sq  = this->x()*this->x();
        const Scalar s2Sq  = this->y()*this->y();
        const Scalar s3Sq  = this->z()*this->z();

        res.coeffRef(0,0) = Scalar(4)*(+s1Sq - s2Sq - s3Sq)+ms2Sq;
        res.coeffRef(0,1) = s1s2 - Scalar(4)*this->z()*ms2;
        res.coeffRef(0,2) = s1s3 + Scalar(4)*this->y()*ms2;
        res.coeffRef(1,0) = s1s2 + Scalar(4)*this->z()*ms2;
        res.coeffRef(1,1) = Scalar(4)*(-s1Sq + s2Sq - s3Sq)+ms2Sq;
        res.coeffRef(1,2) = s2s3 - Scalar(4)*this->x()*ms2;
        res.coeffRef(2,0) = s1s3 - Scalar(4)*this->y()*ms2;
        res.coeffRef(2,1) = s2s3 + Scalar(4)*this->x()*ms2;
        res.coeffRef(2,2) = Scalar(4)*(-s1Sq - s2Sq + s3Sq)+ms2Sq;
        res = res / ps2 / ps2;
        
        return res;
    }

    /** Sets \c *this to be a MRP representing a rotation between
     * the two arbitrary vectors \a a and \a b. In other words, the built
     * rotation represent a rotation sending the line of direction \a a
     * to the line of direction \a b, both lines passing through the origin.
     *
     * \returns a reference to \c *this.
     *
     * Note that the two input vectors do \b not have to be normalized, and
     * do not need to have the same norm.
     */
    template<class Derived>
    template<typename Derived1, typename Derived2>
    inline Derived& MRPBase<Derived>::setFromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b)
    {
        using std::acos;
        using std::tan;

        Vector3 v0 = a.normalized();
        Vector3 v1 = b.normalized();
        Scalar c = v1.dot(v0);

        if (c <= 1 && c >= -1) {
            Vector3 axis = v0.cross(v1);
            axis.normalize();

            this->vec() = axis*tan(acos(c)/Scalar(4.0));
        } else {
            this->vec() << 0., 0., 0.;
        }
        return derived();
    }


    /** Returns a MRP representing a rotation between
     * the two arbitrary vectors \a a and \a b. In other words, the built
     * rotation represent a rotation sending the line of direction \a a
     * to the line of direction \a b, both lines passing through the origin.
     *
     * \returns resulting MRP
     *
     * Note that the two input vectors do \b not have to be normalized, and
     * do not need to have the same norm.
     */
    template<typename Scalar, int Options>
    template<typename Derived1, typename Derived2>
    MRP<Scalar,Options> MRP<Scalar,Options>::FromTwoVectors(const MatrixBase<Derived1>& a, const MatrixBase<Derived2>& b)
    {
        MRP sig;
        sig.setFromTwoVectors(a, b);
        return sig;
    }

    /** \returns the MRP Shadow set of \c *this
     *
     * \sa MRPBase::shadow()
     */
    template <class Derived>
    inline MRP<typename internal::traits<Derived>::Scalar> MRPBase<Derived>::shadow() const
    {
        Scalar n2 = this->squaredNorm();
        if (n2 > Scalar(0))
            return MRP<Scalar>(-this->coeffs() / n2);
        else {
            return MRP<Scalar>(0.,0.,0.);
        }
    }

    /** \returns the MRP [B] matrix of \c *this
     * This is used in
     *
     *  d(sigmda_B/N) = 1/4 [B(sigma_B/N)] omega_BN_B
     *
     * \sa MRPBase::shadow()
     */
    template <class Derived>
    inline typename MRPBase<Derived>::Matrix3 MRPBase<Derived>::Bmat() const
    {
        Matrix3 B;
        Scalar ms2, s1s2, s1s3, s2s3;

        ms2  = Scalar(1) - this->coeffs().squaredNorm();
        s1s2 = this->x()*this->y();
        s1s3 = this->x()*this->z();
        s2s3 = this->y()*this->z();

        B.coeffRef(0,0) = ms2 + Scalar(2)*this->x()*this->x();
        B.coeffRef(0,1) = Scalar(2)*(s1s2 - this->z());
        B.coeffRef(0,2) = Scalar(2)*(s1s3 + this->y());
        B.coeffRef(1,0) = Scalar(2)*(s1s2 + this->z());
        B.coeffRef(1,1) = ms2 + Scalar(2)*this->y()*this->y();
        B.coeffRef(1,2) = Scalar(2)*(s2s3 - this->x());
        B.coeffRef(2,0) = Scalar(2)*(s1s3 - this->y());
        B.coeffRef(2,1) = Scalar(2)*(s2s3 + this->x());
        B.coeffRef(2,2) = ms2 + Scalar(2)*this->z()*this->z();

        return B;
    }

    /** \returns the multiplicative inverse of \c *this
     * Note that in most cases, i.e., if you simply want the opposite rotation,
     * and/or the MRP is normalized, then it is enough to use the conjugate.
     *
     * \sa MRPBase::conjugate()
     */
    template <class Derived>
    inline MRP<typename internal::traits<Derived>::Scalar> MRPBase<Derived>::inverse() const
    {
        return MRP<Scalar>(-this->coeffs());
    }

    /** \returns the conjugate of the \c *this which is equal to the multiplicative inverse
     * if the MRP is normalized.
     * The conjugate of a MRP represents the opposite rotation.
     *
     * \sa MRP2::inverse()
     */
    template <class Derived>
    inline MRP<typename internal::traits<Derived>::Scalar>
    MRPBase<Derived>::conjugate() const
    {
        return MRP<Scalar>(-this->coeffs());
    }

    /** \returns the angle (in radian) between two rotations
     * \sa dot()
     */
    template <class Derived>
    template <class OtherDerived>
    inline typename internal::traits<Derived>::Scalar
    MRPBase<Derived>::angularDistance(const MRPBase<OtherDerived>& other) const
    {
        using std::atan;
        using std::abs;
        MRP<Scalar> d = (*this) * other.conjugate();
        return Scalar(4) * atan( d.norm() );
    }



    //    /** \returns the spherical linear interpolation between the two MRPs
    //     * \c *this and \a other at the parameter \a t in [0;1].
    //     *
    //     * This represents an interpolation for a constant motion between \c *this and \a other,
    //     * see also http://en.wikipedia.org/wiki/Slerp.
    //     */
    //    template <class Derived>
    //    template <class OtherDerived>
    //    MRP<typename internal::traits<Derived>::Scalar>
    //    MRPBase<Derived>::slerp(const Scalar& t, const MRPBase<OtherDerived>& other) const
    //    {
    //        using std::acos;
    //        using std::sin;
    //        using std::abs;
    //        static const Scalar one = Scalar(1) - NumTraits<Scalar>::epsilon();
    //        Scalar d = this->dot(other);
    //        Scalar absD = abs(d);
    //
    //        Scalar scale0;
    //        Scalar scale1;
    //
    //        if(absD>=one)
    //        {
    //            scale0 = Scalar(1) - t;
    //            scale1 = t;
    //        }
    //        else
    //        {
    //            // theta is the angle between the 2 MRPs
    //            Scalar theta = acos(absD);
    //            Scalar sinTheta = sin(theta);
    //
    //            scale0 = sin( ( Scalar(1) - t ) * theta) / sinTheta;
    //            scale1 = sin( ( t * theta) ) / sinTheta;
    //        }
    //        if(d<Scalar(0)) scale1 = -scale1;
    //
    //        return MRP<Scalar>(scale0 * coeffs() + scale1 * other.coeffs());
    //    }

    namespace internal {

        // set from a rotation matrix
        // this maps the [NB] DCM to the equivalent sigma_B/N set
        template<typename Other>
        /*! struct definition */
        struct MRPbase_assign_impl<Other,3,3>
        {
            typedef typename Other::Scalar Scalar; //!< variable
            typedef DenseIndex Index; //!< variable
            template<class Derived> static inline void run(MRPBase<Derived>& sig, const Other& mat)
            {
                Quaternion<Scalar> q;
                Scalar num;

                /* convert DCM to quaternions */
                q = mat;
                num = Scalar(1) + q.w();

                /* convert quaternions to MRP */
                sig.x() = q.x()/num;
                sig.y() = q.y()/num;
                sig.z() = q.z()/num;
            }
        };
        
        // set from a vector of coefficients assumed to be a MRP
        template<typename Other>
        /*! struct definition */
        struct MRPbase_assign_impl<Other,3,1>
        {
            typedef typename Other::Scalar Scalar; //!< variable
            template<class Derived> static inline void run(MRPBase<Derived>& q, const Other& vec)
            {
                q.coeffs() = vec;
            }
        };
        
    } // end namespace internal
    
} // end namespace Eigen

#endif // EIGEN_MRP_H

/// \endcond
